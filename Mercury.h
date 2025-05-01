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
        std::map<std::wstring, unsigned int> tokenToId;
        std::map<unsigned int, std::wstring> idToToken;

        public :

        void learn(const std::string path);
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

    bool inArrayWchar(const std::vector<wchar_t> array, const wchar_t value);
    bool inArrayWstring(const std::vector<std::wstring> array, const std::wstring value);
    bool isAlNum(const wchar_t ch);
    std::wstring trim(const std::wstring text);
    unsigned int wstringToInt(const std::wstring value);
    std::vector<std::wstring> explode(std::wstring str, const wchar_t separator);
    std::vector<std::wstring> getMaxPairs(std::map<std::wstring, unsigned int> pairs);
}
