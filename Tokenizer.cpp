#include "Mercury.h"

void Mercury::Tokenizer::learn(const std::string path)
{
    tokens.clear();
    tokenToId.clear();
    idToToken.clear();

    std::wfstream file(path + "/Mercury/Corpus.txt");
    if(!file)
    {
        std::cout << "Mercury : error, Can't find Corpus.txt" << std::endl;
        exit(-1);
    }

    std::cout << "Mercury : Tokenization ..." << std::endl;

    std::wstring corpus;

    while(1)
    {
        std::wstring line;
        if(!getline(file, line))
        {
            break;
        }

        std::wstring lineFiltered = L"";
        for(const wchar_t ch : line)
        {
            if(isAlNum(ch))
            {
                lineFiltered += ch;
            }
        }

        corpus += lineFiltered;
    }

    file.close();

    //corpus = L"Les chats sont les rois du chapeau";
    //corpus = L"coder coding gamer gaming";

    std::vector<std::wstring> corpus2;

    for(const wchar_t ch : corpus)
    {
        if(ch != L' ' && ch != L'\t')
        {
            std::wstring letter = L"";
            letter += ch;

            //if(!inArrayWstring(corpus2, letter))
            {
                corpus2.push_back(letter);
            }

            tokens[letter]++;
        }
    }

    /*for(const std::wstring c : corpus2)
    {
        std::wcout << c << std::endl;
    }*/

    bool quit = false;

    while(1)
    {
        std::map<std::wstring, unsigned int> pairs;

        for(size_t i = 0 ; i < corpus2.size() - 1 ; i++)
        {
            std::wstring pair = L"";

            pair += corpus2.at(i + 0);
            pair += corpus2.at(i + 1);

            if(pair.length() < 15)
            {
                pairs[pair]++;
            }

            else
            {
                quit = true;
            }
        }

        if(quit)
        {
            break;
        }

        /*for(const auto& kv : pairs)
        {
            std::wcout << kv.first << std::endl;
        }*/

        const std::vector<std::wstring> maxPairs = getMaxPairs(pairs);

        for(size_t i = corpus2.size() - 1 ; i > 0 ; i--)
        {
            std::wstring pair = L"";

            pair += corpus2.at(i - 1);
            pair += corpus2.at(i - 0);

            if(inArrayWstring(maxPairs, pair))
            {
                corpus2.erase(corpus2.begin() + (i - 1));
                corpus2.erase(corpus2.begin() + (i - 1));

                corpus2.insert(corpus2.begin() + (i - 1), pair);

                tokens[pair]++;
            }
        }

        /*std::cout << "__________" << std::endl;
        for(const std::wstring c : corpus2)
        {
            std::wcout << c << std::endl;
        }

        getch();*/
    }

    /*for(const auto &kv : tokens)
    {
        std::wcout << kv.first << std::endl;
    }*/

    //std::cout << tokens.size() << std::endl;

    ///////////////////////////////////////////////////////////////////////////

    std::wofstream file2(path + "/Mercury/Tokens.txt");

    tokenToId[L" "] = 1;
    idToToken[1] = L" ";

    unsigned int id = 2;
    for(const auto &kv : tokens)
    {
        //std::wcout << kv.first << " => " << kv.second << std::endl;
        file2 << L"[" << kv.first << L"|" << kv.second << L"]";

        if(id == tokens.size() - 1)
        {
            file2 << std::endl;
        }

        tokenToId[kv.first] = id;
        idToToken[id] = kv.first;

        id++;
    }

    size_t index = 0;
    for(const auto &kv : tokenToId)
    {
        file2 << L"[" << kv.first << L"|" << kv.second << L"]";

        if(index == tokenToId.size() - 1)
        {
            file2 << std::endl;
        }

        index++;
    }

    index = 0;
    for(const auto &kv : idToToken)
    {
        file2 << L"[" << kv.first << L"|" << kv.second << L"]";

        if(index == idToToken.size() - 1)
        {
            file2 << std::endl;
        }

        index++;
    }

    file2.close();
}

void Mercury::Tokenizer::loadDatas(const std::string path)
{
    std::wifstream file(path + "/Mercury/Tokens.txt");
    if(!file)
    {
        //std::cout << "Error : Tokens.txt not found, please learn the tokenizer before" << std::endl;
        return;
    }

    std::wstring line;
    getline(file, line);

    std::wstring value = L"";
    for(const wchar_t ch : line)
    {
        if(ch != L'[' && ch != L']')
        {
            value += ch;
        }

        if(ch == L']')
        {
            std::wstring token = value;
            std::wstring id = value;

            value = L"";

            token.erase(token.rfind(L"|"));
            id.erase(0, id.rfind(L"|") + 1);

            tokens[token] = wstringToInt(id);
        }
    }

    ///////////////////////////////////////////////

    getline(file, line);
    value = L"";
    for(const wchar_t ch : line)
    {
        if(ch != L'[' && ch != L']')
        {
            value += ch;
        }

        if(ch == L']')
        {
            std::wstring token = value;
            std::wstring id = value;

            value = L"";

            token.erase(token.rfind(L"|"));
            id.erase(0, id.rfind(L"|") + 1);

            tokenToId[token] = wstringToInt(id);
        }
    }

    ///////////////////////////////////////////////

    getline(file, line);
    value = L"";
    for(const wchar_t ch : line)
    {
        if(ch != L'[' && ch != L']')
        {
            value += ch;
        }

        if(ch == L']')
        {
            std::wstring id = value;
            std::wstring token = value;

            value = L"";

            id.erase(id.rfind(L"|"));
            token.erase(0, token.rfind(L"|") + 1);

            idToToken[wstringToInt(id)] = token;
        }
    }

    file.close();
}

std::vector<unsigned int> Mercury::Tokenizer::encode(const std::wstring text)
{
    std::vector<std::wstring> localTokens;

    for(const auto &kv : tokens)
    {
        localTokens.push_back(kv.first);
    }

    bool sorted;
    while(1)
    {
        sorted = true;

        for(size_t i = 0 ; i < localTokens.size() - 1 ; i++)
        {
            const std::wstring token1 = localTokens[i + 0];
            const std::wstring token2 = localTokens[i + 1];

            if(token1.length() < token2.length())
            {
                sorted = false;

                localTokens[i + 0] = token2;
                localTokens[i + 1] = token1;
            }
        }

        if(sorted)
        {
            break;
        }
    }

    /////////////////////////////

    /*for(const std::wstring t : localTokens)
    {
        std::wcout << t << std::endl;
    }*/

    std::vector<unsigned int> res;

    std::vector<std::wstring> words = explode(text, L' ');

    size_t index = 0;
    for(const std::wstring word : words)
    {
        //std::wcout << word << L" => " << std::endl;
        size_t pos = 0;
        while(1)
        {
            bool tokenFound = false;

            for(size_t i = 0 ; i < localTokens.size() ; i++)
            {
                const size_t size = localTokens[i].length();
                const std::wstring subWord = word.substr(pos, size);

                if(subWord == localTokens[i])
                {
                    res.push_back(tokenToId[subWord]);
                    //std::wcout << subWord << std::endl;
                    pos += size;
                    tokenFound = true;
                    break;
                }
            }

            if(!tokenFound)
            {
                pos++;      //Cas token non trouvé
            }

            if(pos >= word.length())
            {
                break;
            }
        }

        if(index < words.size() - 1)
        {
            res.push_back(tokenToId[L" "]);     //Spaces
        }

        index++;
    }

    /////////////////////////////

    return res;
}

std::wstring Mercury::Tokenizer::decode(std::vector<unsigned int> &localTokens)
{
    std::wstring res = L"";

    for(const unsigned int id : localTokens)
    {
        res += idToToken[id];
    }

    return res;
}
