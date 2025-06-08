#include <iostream>
#include <vector>
#include <map>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <conio.h>

namespace Mercury
{
    class Tokenizer
    {
        private :

        std::map<std::wstring, unsigned int> tokens;
        //std::map<std::wstring, unsigned int> tokenToId;
        std::map<unsigned int, std::wstring> idToToken;

        public :

        void learn(const std::string path);
        void getFirstTokens(const std::wstring corpus, unsigned int &id);
        void loadDatas(const std::string path);

        std::vector<unsigned int> encode(const std::wstring text);
        std::wstring decode(std::vector<unsigned int> &localTokens);

        std::map<std::wstring, unsigned int> &getTokens()
        {
            return tokens;
        }
    };

    class ChatBot
    {
        private :

        std::string path = ".";
        Tokenizer tokenizer;

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
}

