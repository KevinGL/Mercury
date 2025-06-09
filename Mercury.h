#include <iostream>
#include <vector>
#include <map>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <cmath>
#include <conio.h>

#define MERCURY_MAX_SIZE_TOKENS 4
#define MERCURY_MAX_SIZE_EMBEDDINGS 4//100
#define MERCURY_SIZE_HIDDEN_LAYER 128
#define MERCURY_MAX_TOKENS_OUTPUT_LAYER 10000

namespace Mercury
{
    struct Neuron
    {
        std::vector<float> weights;
        float bias;
        float output;
        unsigned int indexLayer;
    };

    struct Layer
    {
        std::vector<Neuron> neurons;
    };

    struct Network
    {
        Layer input;
        Layer hidden;
        Layer output;
    };

    class Tokenizer
    {
        private :

        std::map<std::wstring, unsigned int> tokens;
        //std::map<std::wstring, unsigned int> tokenToId;
        std::map<unsigned int, std::wstring> idToToken;

        void getFirstTokens(const std::wstring corpus, unsigned int &id);

        public :

        void learn(const std::string path);
        void loadDatas(const std::string path);
        std::vector<unsigned int> getArrayIds();

        std::vector<unsigned int> encode(const std::wstring text);
        std::wstring decode(std::vector<unsigned int> &localTokens);

        std::map<std::wstring, unsigned int> &getTokens()
        {
            return tokens;
        }

        std::map<unsigned int, std::wstring> &getIds()
        {
            return idToToken;
        }
    };

    class Embedder
    {
        private :

        std::map<unsigned int, std::vector<float>> embeddings;
        std::map<std::vector<float>, unsigned int> embToId;
        Network predictionNetwork;

        void InitNetwork();

        public :

        void learn(const std::string path, Tokenizer &tokenizer);
    };

    class ChatBot
    {
        private :

        std::string path = ".";
        Tokenizer tokenizer;
        Embedder embedder;

        public :

        ChatBot();
        void learn();
        void prompt(const std::wstring text);
    };

    bool isAlNum(const wchar_t ch);
    std::wstring trim(const std::wstring text);
    unsigned int wstringToInt(const std::wstring value);
    std::vector<std::wstring> explode(std::wstring str, const wchar_t separator);
    std::map<std::wstring, unsigned int> getGroupsFromCorpus(const std::wstring corpus, unsigned int groupsSize);
    std::vector<std::wstring> getMaxGroups(std::map<std::wstring, unsigned int> pairs);

    template <typename T>
    bool inArray(const std::vector<T> array, const T value)
    {
        for(const T item : array)
        {
            if(item == value)
            {
                return true;
            }
        }

        return false;
    }

    void softmax(Layer &layer, std::vector<float> &res);
    size_t getIndexMax(std::vector<float> &values);
}

