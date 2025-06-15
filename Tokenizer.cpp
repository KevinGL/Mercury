#include "Mercury.h"

void Mercury::Tokenizer::learn(const std::string path)
{
    tokens.clear();
    idToToken.clear();

    std::wifstream file(path + "/Mercury/Corpus.txt");
    //std::wifstream file(path + "/Mercury/Corpus2.txt");

    if(!file)
    {
        std::cout << "Mercury : Error, " << path << "/Mercury/Corpus.txt does not exit" << std::endl;
        exit(-1);
    }

    std::wstring corpus = L"";

    while(1)
    {
        std::wstring line;

        if(!getline(file, line))
        {
            break;
        }

        corpus += line;// + L'\n';
    }

    file.close();

    unsigned int id = 1;

    getFirstTokens(corpus, id);

    for(size_t i = 2 ; i < MERCURY_MAX_SIZE_TOKENS + 1 ; i++)
    {
        std::map<std::wstring, unsigned int> groups = getGroupsFromCorpus(corpus, i);
        std::vector<std::wstring> maxGroups = getMaxGroups(groups);

        for(const std::wstring group : maxGroups)
        {
            if(tokens.count(group) == 0 && group != L"")
            {
                tokens[group] = id;
                idToToken[id] = group;
                id++;
            }
        }
    }

    std::wofstream file2(path + "/Mercury/Tokens.txt");

    for(const auto& kv : tokens)
    {
        file2 << kv.first << L"|" << kv.second << std::endl;
    }

    file2.close();

    std::wofstream file3(path + "/Mercury/IdToTokens.txt");

    for(const auto& kv : idToToken)
    {
        file3 << kv.first << L"|" << kv.second << std::endl;
    }

    file3.close();
}

void Mercury::Tokenizer::loadDatas(const std::string path)
{
    std::wifstream file(path + "/Mercury/Tokens.txt");
    if(!file)
    {
        //std::cout << "Error : Tokens.txt not found, please learn the tokenizer before" << std::endl;
        return;
    }

    while(1)
    {
        std::wstring line;

        if(!getline(file, line))
        {
            break;
        }

        std::wstring token = line;
        std::wstring id = line;

        token.erase(token.find(L"|"));
        id.erase(0, id.find(L"|") + 1);

        tokens[token] = wstringToInt(id);
    }

    file.close();

    //////////////////

    std::wifstream file2(path + "/Mercury/IdToTokens.txt");
    if(!file2)
    {
        return;
    }

    while(1)
    {
        std::wstring line;

        if(!getline(file2, line))
        {
            break;
        }

        std::wstring id = line;
        std::wstring token = line;

        id.erase(id.find(L"|"));
        token.erase(0, token.find(L"|") + 1);

        idToToken[wstringToInt(id)] = token;
    }

    file2.close();
}

std::vector<unsigned int> Mercury::Tokenizer::encode(const std::wstring text)
{
    std::vector<unsigned int> res;

    if(text == L"")
    {
        return res;
    }

    std::vector<std::wstring> words;

    std::wstring word = L"";
    for(size_t i = 0 ; i < text.length() ; i++)
    {
        if(text[i] != L' ')
        {
            word += text[i];
        }

        else
        {
            words.push_back(word);
            words.push_back(L" ");

            word = L"";
        }
    }
    words.push_back(word);

    for(const std::wstring w : words)
    {
        size_t offset = 0;
        size_t loops = 0;

        while(1)
        {
            for(size_t size = MERCURY_MAX_SIZE_TOKENS ; size > 0 ; size--)
            {
                if(size > w.length() - offset)
                {
                    size = w.length() - offset;
                }

                if(size == 0)
                {
                    break;
                }

                std::wstring token = w;
                //std::wcout << token << L" offset=" << offset << L" offset+size=" << offset + size << std::endl;
                token.erase(offset + size);
                token.erase(0, offset);

                /*std::wcout << offset << L" " << size << L" => \"" << token << L"\" => " << tokens.count(token) << std::endl;
                getch();*/

                if(tokens.count(token) == 1)
                {
                    res.push_back(tokens[token]);
                    offset += size;
                    loops = 0;

                    break;
                }
            }

            loops++;

            if(offset >= w.length() || loops > 10)
            {
                break;
            }
        }
    }

    return res;
}

std::wstring Mercury::Tokenizer::decode(std::vector<unsigned int> &localTokens)
{
    std::wstring res = L"";

    for(const unsigned int localToken : localTokens)
    {
        if(idToToken.count(localToken) == 1)
        {
            res += idToToken[localToken];
        }
    }

    return res;
}

std::vector<unsigned int> Mercury::Tokenizer::getArrayIds()
{
    std::vector<unsigned int> res;

    for(const auto& kv : tokens)
    {
        res.push_back(kv.second);
    }

    return res;
}
