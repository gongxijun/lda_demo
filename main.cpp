#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <chrono>
#include<cmath>
#include <map>
#include <random>
#include <string>
#include <set>
#include <sstream>
#include <climits>
#include <stdexcept>
#include "cppjieba/Jieba.hpp"
#include "my_rand.h"
//#define debug


const char* const DICT_PATH = "thirdparty/cppjieba/dict/jieba.dict.utf8";
const char* const HMM_PATH = "thirdparty/cppjieba/dict/hmm_model.utf8";
const char* const USER_DICT_PATH = "thirdparty/cppjieba/dict/user.dict.utf8";
const char* const IDF_PATH = "thirdparty/cppjieba/dict/idf.utf8";
const char* const STOP_WORD_PATH = "thirdparty/cppjieba/dict/stop_words.utf8";

#ifdef _MSC_VER

typedef __int32 int32_t;
typedef unsigned __int32 uint32_t;
typedef __int64 int64_t;
typedef unsigned __int64 uint64_t;

#endif





class simpleLDA
{
public:
    // estimate LDA model using Gibbs sampling
    simpleLDA();
    ~simpleLDA();
    std::set<std::string> load_stop_words(const char * path_file);  //加载暂停词
    void load_data(const char* path_file ); //加载数据

    int sampling(unsigned m);
    int init_train();
    int train();
    int test(const char * doc);
    bool isvalid(const char * word );
    int save_top_words(std::basic_string<char, std::char_traits<char>, std::allocator<char>> dst_path);

protected:
    /******* dataset info *******/

    std::map<unsigned int, std::string> id2word;			// word map [int => string]
    std::map<std::string, unsigned int> word2id;			// word map [string => int]
    std::set<std::string> stopwords;
    std::vector<std::vector<unsigned int  > > docs ; // all docs

    /****** Model Parameters ******/
    unsigned N;							// Number of documents
    unsigned M; 							// Number of words in dictionary
    unsigned short K; 							// Number of topics


    /****** Model Hyper-Parameters ******/
    double alpha, alphaK;					// per document Topic proportions dirichlet prior
    double beta, Vbeta;				// Dirichlet language model

    /******* Train variable *******/
    unsigned int iterationNum;  // max iteration for train

    /****** Model variables ******/
    unsigned short ** Z;						// topic assignment for each word
    unsigned   ** n_zw;					// number of times word w assigned to topic k
    unsigned ** n_dz; //sparse representation of n_mk: number of words assigned to topic k in document m
    unsigned * n_z;						// number of words assigned to topic k = sum_w n_wk = sum_m n_mk

    /**** template model variables  ****/
     double * p;

    /**** model save parameters ***/
    unsigned int step_size;

    /******* random number generator ******/
    xorshift128plus rng_;


    /****** Functions to update sufficient statistics ******/
    inline int add_to_topic(unsigned doc_n, unsigned word_id, unsigned short topic)
    {
        ++n_z[ topic ];
        ++n_zw[ word_id ][ topic ];
        ++n_dz[ doc_n ][ topic ];

        return 0;
    }

    inline int remove_from_topic(unsigned doc_n, unsigned word_id, unsigned short topic)
    {
        --n_z[topic];
        --n_zw[word_id][topic];
        --n_dz[doc_n][topic];

        return 0;
    }



};

void simpleLDA::load_data(const char* path_file ){ //加载数据

    cppjieba::Jieba jieba(DICT_PATH,
                          HMM_PATH,
                          USER_DICT_PATH,
                          IDF_PATH,
                          STOP_WORD_PATH);
    std::vector<std::string> words;


    std::ifstream fin;

    if (path_file == NULL){
        //check path_file is empty
        std::cout<<"dataset can not find ! "<<std::endl;
    }

    std::cout<<"load the dataset now .... :   "<<path_file << " ."<<std::endl;

    id2word.clear();
    word2id.clear();

    fin.open(path_file , std::ios::in);
    std::string temp;
    auto max_indx=0;
    while(!fin.eof()){
        std::getline(fin ,temp,'\n');
        jieba.Cut( temp , words, true);  //HMM cut the sentence
        std::vector<unsigned int> doc;

        for(auto word : words){
            if( isvalid(word.c_str()) ) {
                auto indx = static_cast<unsigned int>(word2id.size() + 1);
                if (word2id.find(word) == word2id.end()) {
                    word2id[word] = indx;
                    id2word[indx] = word;
                }
                doc.push_back(word2id[word]);
                max_indx=indx;
#ifdef debug
                std::cout << word << std::endl;
#endif
            }
        }
        //std::cout << limonp::Join(words.begin(), words.end(), "/") << std::endl;

        docs.push_back(doc);
    }
    std::cout<<"data loaded completed ! total words"<<max_indx<<std::endl;
    return ;
}

std::set<std::string> simpleLDA::load_stop_words(const char *path_file) {
    std::ifstream fin;

    if (path_file == NULL){
        //check path_file is empty
        std::cout<<"stop_words can not find ! "<<std::endl;
    }
    std::cout<<"load the stop_words now .... :   "<<path_file << " ."<<std::endl;
    stopwords.clear();
    fin.open(path_file , std::ios::in);
    std::string temp;
    while(!fin.eof()){
        std::getline(fin ,temp,'\n');
        stopwords.insert(temp);
    }
    return stopwords;
}


simpleLDA::simpleLDA() {
    this->K=30;
    this->alpha = 50.0;
    this->alphaK = 0;
    this->beta = 0.1;
    this->Vbeta = 0;
    this->iterationNum =20000;
    this->step_size =50;

    this->Z = NULL;
    this->n_z = NULL;
    this->n_dz = NULL;
    this->n_zw = NULL;
}
simpleLDA::~simpleLDA() {

    if(this->Z){
        for(unsigned n = 0 ; n < N ; ++n){
            delete Z[n];
        }
        delete this->Z;
    }

    if(n_zw){
        for (unsigned w = 0; w < M; w++)
        {
            delete n_zw[w];
        }
        delete n_zw;
    }

    if(n_z){
        delete n_z;
    }

    if(n_dz){
        for(unsigned n =0 ; n< N  ; ++n){
            delete this->n_dz[n];
        }
        delete  this->n_dz;
    }
}


int simpleLDA::init_train() {
    this->N = static_cast<unsigned int>(docs.size());
    this->M = static_cast<unsigned int>(word2id.size());
    alphaK = alpha/K;
    Vbeta = M * beta;

    // allocate heap memory for model variables
    n_zw = new unsigned*[M];
    for (unsigned w = 0; w < M; w++)
    {
        n_zw[w] = new unsigned[K];
        for (unsigned short k = 0; k < K; k++)
        {
            n_zw[w][k] = 0;
        }
    }

    n_z = new unsigned[K];
    for (unsigned short k = 0; k < K; k++)
    {
        n_z[k] = 0;
    }


    this->n_dz = new unsigned*[N];
    for(unsigned n =0 ; n< N  ; ++n){
        this->n_dz[n] = new unsigned [K];
        for(unsigned k =0 ; k <K ; ++k){
            this->n_dz[n][k]=0;
        }
    }

    this->Z = new unsigned short*[N];
    for(unsigned n = 0 ; n < N ; ++n){
        Z[n] = new unsigned short [M];
        for(unsigned w =0 ; w<docs[n].size() ;++w){
            unsigned word_id = docs[n][w]-1;
            unsigned short topic = static_cast<unsigned short>(rng_.rand_k(K));
            Z[n][word_id] = topic;
            ++n_zw[ word_id ][topic];
            ++n_z[topic];
            ++n_dz[n][topic];
        }
    }
    this->p=new double[K];
    return 0;
}

bool simpleLDA::isvalid(const char * word ){ //判断是否是一个合格的关键字，不能有魔法数字，不能有暂停词，长度大于1

    uint wlen = strlen(word);

    if (word ==NULL || wlen<2||stopwords.end() != stopwords.find(word)){
        return false;
    }

    for(int i =0 ; i<wlen ; ++i ){ //判断是否含有数字
        if (isdigit(word[i]))
            return false;
    }

    return true;
}


int simpleLDA::train() {
    //训练过程
    init_train();
    std::cout << "Testing at start" << std::endl;

    std::chrono::high_resolution_clock::time_point ts, tn; ts=tn;
    std::cout << "Sampling " << this->iterationNum << " iterations!" << std::endl;
    for (unsigned iter = 0; iter < this->iterationNum; ++iter)
    {
        if (iter % this->step_size == 0)
        {
            //time_ellapsed.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(tn - ts).count());
            //test();
            // saving the model
            std::cout << "Saving the model at iteration " << iter << "..." << std::endl;
            //save_model(iter);
            save_top_words("/Users/sina/github/lda_demo/model/single_lda"+std::to_string(iter)+".topwords");
            //test
            const char *doc_demo="信息时报讯 （记者 刘军 通讯员 龙翊雯 张毅涛） 可美白、抗衰老的“胎盘素”，功能实为主治“更年期障碍、乳汁分泌不全”，搽在脸上麻醉的“麻膏”竟是用于男性性功能保健的药物……整容差点成了毁容！　　近日，广州天河警方在“3+2”专项打击整治行动中端掉一个“地下”美容窝点，抓获8名犯罪嫌疑人，现场缴获境外生产的医疗器械、假药一批。经查，该窝点在无任何资质的情况下进行注射美容，涉嫌非法经营医疗器材、非法销售假药等犯罪，涉案金额达500余万元人民币。　　案例　　女子轻信200元微整形套餐　　陈 女士年近40，是位爱美人士，渴望拥有一张光滑白皙的“明星脸”。一天，她在浏览网站时看到一则美容公司广告，其宣称为港澳最大的微整形美容连锁机构旗舰 店，拥有世界顶级美容机器“嫩肤皇后”。顾客只需花200元，即可体验到安全无创的“微整形”。这让陈女士心动不已，6月10日，她自备了一盒网购的“艺 人玻尿酸”，来到了位于体育西路一写字楼内的广州市某美容仪器有限公司。　　陈女士看到，该公司办公室的外间用于办公和产品陈列，内间摆放 着储物柜、一张简易手术床和一台“嫩肤皇后”美容机。自称阿长和阿娇的美容顾问指引陈女士躺在床上，没有任何消毒程序，两人随手戴上一副透明薄膜手套，开 始了“微整形”操作。阿娇先把一种“麻膏”涂抹在陈女士脸上，声称这是可以减轻注射疼痛的进口麻药。随后，阿长将玻尿酸溶液装入一次性注射器内，并安装在 美容机上。一会儿，这盒药水将在美容机的控制下，通过针头注入陈女士的面部肌肤里。　　乱象　　美容“洋货”全为无证假药　　其 实在6月初，这家办公室就引起了天河警方注意。民警发现，这家公司并无任何资质，却暗地开展医疗美容活动。6月10日，办案民警突袭上址，抓获冯某 （男，29岁，河南省人）等6名犯罪嫌疑人，并查获境外生产的“肉毒杆菌毒素A型”等5种药品、DQ机器等3种医疗器械一批。　　经鉴定，这些“洋货”没有取得我国医疗器械、药品的注册许可，全部为非法产品和假药。同时该公司不具备医疗美容资质，操作人员也没有执业医师资质。　　一 种“补湿嫩白针”产品的警告说明称，“仅适用于真皮内注射。切勿注入血管内……注入血管，可能会导致血管堵塞，相应组织缺血和坏死等”。不过，这些使用警 示完全被漠视，犯罪嫌疑人交代，操作美容整形机器的都是普通销售人员，其在三天培训后即上岗，有的只有初中文化程度。　　黑幕　　拆分零部件走私入境销售　　这些非法“洋货”从何而来？犯罪嫌疑人交代，该团伙伙同他人从国外购得产品后，将其零部件拆开伪装，偷偷走私进入国内，存放在珠海的一间仓库里，然后通过物流运送到广州或直接寄给客户，形成一条“境外到港口再到内陆”的购销链条。　　经查，2013年以来，该团伙通过各大展会、网络等渠道推销产品，销给美容院、私人诊所、美容项目公司等，遍及全国18个省市。产品单价从百元至万元不等。　　目前，犯罪嫌疑人冯某已被天河警方依法逮捕，犯罪嫌疑人张某等7人已被天河警方依法采取强制措施，此案仍在进一步审查中。";
            this->test(doc_demo);
        }

         std::cout << "Iteration " << iter << " ..." << std::endl;
        ts = std::chrono::high_resolution_clock::now();

        // for each document
        for (unsigned n = 0; n < this->N; ++n)
            this->sampling(n);

        tn = std::chrono::high_resolution_clock::now();
        //std::cout << "\rLDA: M = " << M << ", K = " << K << ", V = " << V << ", alpha = "
        //	  << alpha << ", beta = " << beta << std::endl; //
    }
  //  test();
    std::cout << "Gibbs sampling completed!" << std::endl;
    std::cout << "Saving the final model!" << std::endl;
//save_model(-1);

    return 0;

}

int simpleLDA::sampling(unsigned n) {
    //采样
    for(unsigned w = 0 ; w < docs[n].size() ; ++w ){
        unsigned  cur_word = docs[n][w]-1;
        unsigned short topic = this->Z[n][cur_word];
        unsigned pre_topic = topic;
        this->remove_from_topic(n,cur_word,topic);
        // 采用积累法进行多项式抽样
        double temp = 0;
        for (unsigned short k = 0; k < this->K; k++)
        {
#ifdef debug
            std::cout<<n_dz[n][k]<<std::endl;
            std::cout<<n_zw[w][k]<<std::endl;
            std::cout<<n_z[k]<<std::endl;
#endif

            temp += ((n_dz[n][k] + alphaK)*(n_zw[cur_word][k] + beta)) / (n_z[k] + Vbeta);
            this->p[k] = temp;
        }
        //对于非标准话的p[]进行样本缩小
        double u = rng_.rand_double() * temp;

        // 找到比u小的第一个topic
        topic = static_cast<unsigned short>(std::lower_bound(p, p + K, u) - p);

        // 添加当前的样本
        add_to_topic( n, cur_word, topic );
        this->Z[n][cur_word] = topic;
    }
    return 0;
}


int simpleLDA::test(const char * doc_context) {

    cppjieba::Jieba jieba(DICT_PATH,
                          HMM_PATH,
                          USER_DICT_PATH,
                          IDF_PATH,
                          STOP_WORD_PATH);
    std::vector<std::string> words;
    jieba.Cut( doc_context , words, true);  //HMM cut the sentence
    std::vector<unsigned int> doc;

    for(auto word : words){
        if( isvalid(word.c_str()) ) {
            auto indx = static_cast<unsigned int>(word2id.size() + 1);
            if (word2id.find(word) != word2id.end()) {
                doc.push_back(word2id[word]);
            }
        }
    }
    //初始化样本
    unsigned short * test_Z = new unsigned short [M];
    unsigned * test_ndz = new unsigned[K];
    for(unsigned w =0 ; w<doc.size() ;++w){
        unsigned word_id = doc[w]-1;
        unsigned short topic = static_cast<unsigned short>(rng_.rand_k(K));
        test_Z[word_id] = topic;
        ++test_ndz[topic];
    }
    //采样
    for(unsigned w = 0 ; w < doc.size() ; ++w ){
        unsigned  cur_word = doc[w]-1;
        // 采用积累法进行多项式抽样
        unsigned short topic = test_Z[cur_word];
        unsigned pre_topic = topic;
        --n_z[topic];
        --n_zw[cur_word][topic];
        --test_ndz[topic];
        // 采用积累法进行多项式抽样
        double temp = 0;
        for (unsigned short k = 0; k < this->K; k++)
        {
            temp += ((test_ndz[k] + alphaK)*(n_zw[cur_word][k] + beta)) / (n_z[k] + Vbeta);
            this->p[k] = temp;
        }
        //对于非标准话的p[]进行样本缩小
        double u = rng_.rand_double() * temp;
        // 找到比u小的第一个topic
        topic = static_cast<unsigned short>(std::lower_bound(p, p + K, u) - p);

        --n_z[pre_topic];
        --n_zw[cur_word][pre_topic];
        ++test_ndz[topic];
        test_Z[w]=topic;

    }
    //统计概率
    int sum_topic=0;
    for(int k=0 ; k<K ; k++){
        sum_topic+=test_ndz[k];
    }

    for(int k=0 ; k<K ; k++){
        std::cout<<"Topic " << k << "th  the  prob:  "<<(test_ndz[k]+0.01)/(sum_topic+0.01)<<std::endl;
    }
    delete [] test_Z;
    delete [] test_ndz;
    return 0;

}


int simpleLDA::save_top_words(std::basic_string<char, std::char_traits<char>, std::allocator<char>> dst_path) {

    std::ofstream fout(dst_path);
    if (!fout)
        throw std::runtime_error( "Error: Cannot open file to save: " + std::string(dst_path) );

    std::map<unsigned, std::string>::const_iterator it;

    for (unsigned short k = 0; k < K; k++)
    {
        std::vector<std::pair<unsigned, unsigned> > words_probs(M);
        std::pair<unsigned, unsigned> word_prob;
        for (int w = 0; w < M; w++)
        {
            word_prob.first = static_cast<unsigned int>(w);
            word_prob.second = this->n_zw[w][k];
            words_probs[w] = word_prob;
        }

        // quick sort to sort word-topic probability
        std::sort(words_probs.begin(), words_probs.end(), [](std::pair<unsigned, unsigned> &left, std::pair<unsigned, unsigned> &right){return left.second > right.second;});

        fout << "Topic " << k << "th:" << std::endl;
        for (unsigned i = 0; i < 15; i++)
        {
            it = id2word.find(words_probs[i].first);
            if (it != id2word.end())
                fout << "\t" << it->second << "   " << words_probs[i].second << std::endl;
        }
    }

    fout.close();
    std::cout << "twords done" << std::endl;

    return 0;
}




int main(int argc , char ** argv){
    simpleLDA *simple_lda = new simpleLDA();
    simple_lda->load_stop_words("/Users/sina/github/lda_demo/data/stopwords.txt");
    simple_lda->load_data("/Users/sina/github/lda_demo/data/dataset_cn.txt");  //init word2id , id2word , docs
    simple_lda->train();
    simple_lda->save_top_words("/Users/sina/github/lda_demo/model/single_lda.topwords");
    delete simple_lda;
    return 0;
}