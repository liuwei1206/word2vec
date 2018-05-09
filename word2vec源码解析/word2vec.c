//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers，浮点数的精度

//词汇中每个word对应的结构体
struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

char train_file[MAX_STRING], output_file[MAX_STRING];                                                  //训练源文件和输出文件
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];                                         //保存词汇文件和读词汇文件
struct vocab_word *vocab;                                                                              //词汇表，所有单词信息
//参数依次为: 文件保存格式， cbow或者skip? debug_mode? 窗口大小? 最小词频? 训练线程数? min_reduce是最小裁剪次数
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1; 
int *vocab_hash;                                                                                       //词汇表hash值？
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;                                    //最大词汇值，当前词汇值，layer1大小？
//train_words? word_count_actual? 迭代次数? 文件大小? 种类?
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;                
real alpha = 0.025, starting_alpha, sample = 1e-3;                                                     //学习率，开始学习率，下采样?
//syn0: 词向量的初始化; syn1: 用Hierarchical Softmax训练; syn1neg: 用负采样训练, expTable: 预先计算好的sigmoid估计值
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;                                                                                         //计时器

int hs = 0, negative = 5;                             
const int table_size = 1e8;
int *table;

//生成负采样的概率表
void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  //pow(x, y)计算x的y次方;train_words_pow表示总的词的概率，不是直接用每个词的频率，而是频率的0.75次方幂
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);  
  i = 0;
  d1 = pow(vocab[i].cn, power) / train_words_pow;
  //每个词在table中占的小格子数是不一样的，频率高的词，占的格子数显然多
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (double)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
//从文件中读取单个单词，假设单词之间通过空格或者tab键或者EOL键进行分割的
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);                                             //读一个词
    if (ch == 13) continue;                                      //如果是换行符                                  
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {           //代表一个单词结束的边界
      if (a > 0) {                                               //如果读到了单词但是遇到了换行符，
        if (ch == '\n') ungetc(ch, fin);                         //退回到流中
        break;
      }
      if (ch == '\n') {                                          //仅仅读到了换行符
        strcpy(word, (char *)"</s>");                            //将</s>赋予给word
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words   //截断
  }
  word[a] = 0;                                                   //最后一个字符是'\0'
}

// Returns hash value of a word
//返回一个词对应的hash值
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
//开放地址发得到词的位置
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);                                     //获得索引
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;                                     //开放定址法
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];                     
  ReadWord(word, fin);                                   //从文件流中读取一个单词
  if (feof(fin)) return -1;
  return SearchVocab(word);                              //返回对应的词汇表中索引
}

// Adds a word to the vocabulary
//将word加入到词汇表
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;                                 //规定每个word不超过MAX_STRING个字符
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);                                         //结构体的word词
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed                                                //动态扩展内存
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;                                                     //词汇量加上1000
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;           //得到word实际对应的hash值
  vocab_hash[hash] = vocab_size - 1;                                            //通过hash值获得word在vocab中索引
  return vocab_size - 1;                                                        //返回单词对应索引
}

// Used later for sorting by word counts
//构造一个比较器，用来排序，降序
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
	//频率低于一定程度的词会被抛弃掉
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
	  //因为排序之后顺序打乱，会重新计算一次hash值
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  //重新规划内存大小
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
//对于频率小于min_reduce的词将会被裁剪掉
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  //仅仅一个数组就实现了裁剪过程
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  vocab_size = b;
  //重新设置hash值
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;                                                             //每次裁剪之后都会提高最低频率数
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  //分配的空间大小为，(vocab_size * 2 + 1) * sizeof(long long),因为hufuman树的特性，所以总结点数是2 * n + 1, 其中n是节点数, 此处应该有错误，是2 * n - 1才对
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));                  //节点对应频率
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));                 //记录每个节点是左节点还是右节点
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));            //父节点位置
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  //后面的设为无穷
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  //如同天才般的代码，一次遍历就构造好了huffuman树, ##注意,这个a还代表了一种顺序，所有count值由小到大的顺序##
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2',注意vocab中的词是已经按照cn排好序的了,是按照降序排列的
	//pos1表示取最原始的词对应的词频,而pos2表示取合并最小值形成的词频
	//连续两次取，两次取的时候代码操作时一模一样的
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;                   //记录好合并形成的父节点的位置
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;                                     //左为0,右为1
  }
  // Now assign binary code to each vocabulary word
  // 建好了hufuman树之后，就需要分配code了，注意这个hufuman树是用一个数组来存储的，并不是我们常用的指针式链表
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];                                 //对于每个节点，自底向上得到code值，通过每个节点的binary来实现
      point[i] = b;                                        //point记录节点到根节点经过的节点的路径
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
    vocab[a].codelen = i;                                  //记录词对应的码值的长度
    vocab[a].point[0] = vocab_size - 2;                    //最大值作为根节点
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];                  //倒序过来，自顶向下
      vocab[a].point[i - b] = point[b] - vocab_size;       //注意这个索引对应的是huffman树中的非叶子节点，对应syn1中的索引
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

//整合上面的文件操作
void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;    //hash值初始为-1
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");                              //将'</s>'添加到词汇表，换行符就是用这个表示
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);                                     //查找该词的位置
    if (i == -1) {                                             //还未加入到词汇表                   
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;                                      //已经加入到词汇表
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();     //裁剪词操作
  }
  SortVocab();                                                 //排序
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

//保存学习到的词汇文件表
void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

//从已有的词汇文件中直接读取，不用临时去学习
void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {                                                //判断文件是否存在
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;         //vocab_hash值默认为-1
  vocab_size = 0;
  while (1) {                                                       //不停读取，直到文件末尾
    ReadWord(word, fin);                                            //从文件流中读取一个单词到word中
    if (feof(fin)) break;
    a = AddWordToVocab(word);                                       //将单词加入到词汇表            
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);                        //读取词频到vocav.cn中，换行符                    
    i++;
  }
  SortVocab();                                                      //排序词汇表
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);                                          //将读取指针定位到文件尾部
  file_size = ftell(fin);                                           //得到离头部偏离值，获取文件大小
  fclose(fin);
}

//初始化网络
void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  //为syn0分配内存，对齐的内存，大小为vocab_size * layer1_size * sizeof(real),也就是每个词汇对应一个layer1_size的向量
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  
  //如果采用huffman softmax构造，那么需要初始化syn1，大小为vocab_size * layer1_size * sizeof(real)，每个词对应一个
  if (hs) {
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1[a * layer1_size + b] = 0;
  }
  
  //如果采用负采样进行训练，那么久初始化syn1neg，大小为vocab_size * layer1_size * sizeof(real)，每个词对应一个
  if (negative>0) {
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1neg[a * layer1_size + b] = 0;
  }
  
  //对syn0中每个词对应的词向量进行初始化
  for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
  }
  
  //构建huffman softmax需要的哈夫曼树
  CreateBinaryTree();
}

//模型训练的线程
void *TrainModelThread(void *id) {
  long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label, local_iter = iter;
  unsigned long long next_random = (long long)id;
  real f, g;
  clock_t now;
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  FILE *fi = fopen(train_file, "rb");
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  while (1) {
	//每训练10000个词时，打印已训练数占所有需要训练数比例，以及打印训练时间；然后更新学习率
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
         word_count_actual / (real)(iter * train_words + 1) * 100,
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    if (sentence_length == 0) {
      while (1) {
        word = ReadWordIndex(fi);                                                   //得到词在词汇表中对应的索引
        if (feof(fi)) break;                                                        //
        if (word == -1) continue;
        word_count++;                                                               //句子总的次数
        if (word == 0) break;                                                       //遇到换行符，则直接跳出来，第一个词'</s>'代表换行符
        // The subsampling randomly discards frequent words while keeping the ranking same
		//下采样随机丢弃频繁的单词，同时保持排名相同，随机跳过一些词的训练
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
		  //频率越大的词，对应的ran就越小，越容易被抛弃，被跳过
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word;                                                //当前句子包含的词，存的是索引
        sentence_length++;                                                          //句子实际长度，减去跳过的词
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }
    word = sen[sentence_position];                                                  //
    if (word == -1) continue;
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
	//b用来确定窗口的起始位置，也就是中心词word的上下文并不一定是左右取相同个词，但总数还是2 * window大小
    b = next_random % window;                                                       
    if (cbow) {  //train the cbow architecture
      // in -> hidden
      cw = 0;
	  //随机取一个词word，然后计算该词上下文词对应的向量的各维度之和
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];                                                         //获得senten中第c个词的索引
        if (last_word == -1) continue;
		//注意syn0是一维数组，不是二维的，所以通过last_word * layer1_size来定位某个词对应的向量位置, last_word表示上下文中上一个词
        for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];  //neu1表示映射层向量，上下文累加平均 
        cw++;
      }
      if (cw) {
		//上下文表示是所有词对应词向量的平均值
        for (c = 0; c < layer1_size; c++) neu1[c] /= cw;
		//注意hs模式下，syn1存的是非叶子节点对应的向量，并不是词汇表中的词对应的另一个向量；而negative模型下，syn1neg存的是词的另一个向量，需要注意
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;
          // Propagate hidden -> output, 传播过程
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];             //注意syn1也是一维数组，不同词对应的位置需要偏移量l2确定
          if (f <= -MAX_EXP) continue;                                               //当f值不属于[-MAX_EXP, MAX_EXP]
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];  //查看f属于第几份，((f + MAX_EXP) / (2 * MAX_EXP)) * EXP_TABLE_SIZE
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;                                 //需要推导,得到这个梯度比例
          // Propagate errors output -> hidden
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];            //这个部分才是最终梯度值
          // Learn weights hidden -> output
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];             //加上梯度值，更新syn1
        }
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {                                                              //一个正样本
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;        //随机挑选一个负样本，负样本就是除中心词以外的所有词
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;            //如果target为0，这个等式保证不为0
            if (target == word) continue;                                            //正样本则跳过
            label = 0;
          }
          l2 = target * layer1_size;                                                 //syn1neg是一维数组，某个词需要先计算偏移量
          f = 0;
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];          //负采样实际会为每个词生成两个向量
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
        }
        // hidden -> in
		//更新输入层的词向量
        for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = sen[c];
          if (last_word == -1) continue;
          for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
        }
      }
    } else {  //train skip-gram
	  //还是保证一个2 * window大小上下文，但是中心词左右并不一定刚好都是window个，根据b确定
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;                          //c表示上下文的当前遍历位置
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];                                          //用来记录上一个词
        if (last_word == -1) continue;                               //如果词不在词汇表中，则直接跳过
        l1 = last_word * layer1_size;                                //偏移量，因为syn0是一维数组，每个词对应的向量需要先偏移前面的词对应向量
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        // HIERARCHICAL SOFTMAX 
		//不需要像cbow一样求平均
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;                   //
          // Propagate hidden -> output
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1];
        }
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {                                                         //正样本
            target = word;
            label = 1;
          } else {                                                              //负样本
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size;                                            //偏移量
          f = 0;
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];//
          if (f > MAX_EXP) g = (label - 1) * alpha;                             //计算梯度
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];    //完整梯度
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];//更新
        }
        // Learn weights input -> hidden
		//更新输入层权重
        for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
      }
    }
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

//训练模型
void TrainModel() {
  long a, b, c, d;
  FILE *fo;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;                                                                         //设置学习率
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();                       //获得词汇表，如果已经有直接读，否则学
  if (save_vocab_file[0] != 0) SaveVocab();
  if (output_file[0] == 0) return;                                                                //必须有输出文件参数
  InitNet();                                                                                      //初始化网络参数
  if (negative > 0) InitUnigramTable();                                                           //如果是使用负采样，那么需要负采样概率表
  start = clock();                                                                                //计时器
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  fo = fopen(output_file, "wb");
  if (classes == 0) {                                                                             //classes判断是否使用kmean聚类，为0表示否
    // Save the word vectors
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
      else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
      fprintf(fo, "\n");
    }
  } else {
    // Run K-means on the word vectors
	//类别中心数，迭代次数，
    int clcn = classes, iter = 10, closeid;
    int *centcn = (int *)malloc(classes * sizeof(int));                                          //每个中心点拥有的词数量
    int *cl = (int *)calloc(vocab_size, sizeof(int));                                            //每个词所属类别标签
    real closev, x;
    real *cent = (real *)calloc(classes * layer1_size, sizeof(real));                            //聚类中心，注意是用一维数组表示，每个中心需要通过偏移量来定位
    for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;                                           //初始化每个词所属类别
    for (a = 0; a < iter; a++) {                                                                 //开始训练
      for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;                                      //初始化中心点位置
      for (b = 0; b < clcn; b++) centcn[b] = 1;                                                  //初始化每个中心点拥有的词的数量
      //求每个中心点每个维度值的总和，等于所有属于这个类别的词向量的相应维度相加
	  for (c = 0; c < vocab_size; c++) {
        for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
        centcn[cl[c]]++;                                                                         //所包含词的数量+1
      }
	  //对于每一个类别，需要更新中心点各维度值，就是总和平均
      for (b = 0; b < clcn; b++) {                                                               
        closev = 0;
        for (c = 0; c < layer1_size; c++) {                                                       //遍历每个维度
          cent[layer1_size * b + c] /= centcn[b];                                                 //每个词每个维度平均
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];                        //求每个中心点的模的平方
        }
        closev = sqrt(closev);                                                                    //每个中心点模
        for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;                    //归一化处理
      }
	  //更新每个词所属的类别，看离哪个中心点最近就归为相应的类别
      for (c = 0; c < vocab_size; c++) {
        closev = -10;                                                                             //记录离最近中心点距离
        closeid = 0;                                                                              //记录最近的类别id
        for (d = 0; d < clcn; d++) {
          x = 0;
          for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }
    // Save the K-means classes
    for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
    free(centcn);
    free(cent);
    free(cl);
  }
  fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");                                        //word vector估测工具包         
    printf("Options:\n");
    printf("Parameters for training:\n");               
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
    return 0;
  }
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);                    //词向量维度
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);                   //训练文件，语料库文件
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);         //词汇表保存文件
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);         //已有词汇表文件
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);                    //是否打印信息，大于1表示打印，默认2
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);                       //词向量结果以文本后者二进制保存，1是二进制，0是文本，默认
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);                           //cbow或skip，1表示cbow
  if (cbow) alpha = 0.05;
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);                         //学习率
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);                 //词向量保存文件
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);                       //窗口大小
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);                       //下采样率，即下采样阀值
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);                               //huffman softmax，默认0，0表示不用
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);                   //负采样大小，0表示不用，默认5，一般3到10
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);                 //训练线程数， 默认12
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);                           //训练迭代数，默认5
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);                 //最小频率， 默认5
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);                     //聚类中心数，默认0
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));       //初始化expTable，近似逼近sigmoid(x)值，x区间为[-MAX_EXP, MAX_EXP]，分成EXP_TABLE_SIZE份
  //将[-MAX_EXP, MAX_EXP]分成EXP_TABLE_SIZE份
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  return 0;
}
