#include "Mercury.h"

void Mercury::Tokenizer::learn(const std::string path)
{
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

    for(size_t i = 2 ; i < 5 ; i++)
    {
        std::map<std::wstring, unsigned int> groups = getGroupsFromCorpus(corpus, i);
        std::vector<std::wstring> maxGroups = getMaxGroups(groups);

        for(const std::wstring group : maxGroups)
        {
            if(tokens.count(group) == 0)
            {
                tokens[group] = id;
                idToToken[id] = group;
                id++;
            }
        }
    }

    /*for(const auto& kv : tokens)
    {
        std::wcout << L"\"" << kv.first << L"\" => " << kv.second << std::endl;
    }*/

    /*for(const auto& kv : idToToken)
    {
        std::wcout << kv.first << L" => " << kv.second << std::endl;
    }*/

    //std::cout << tokens.size() << std::endl;

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

    //

    file.close();
}

std::vector<unsigned int> Mercury::Tokenizer::encode(const std::wstring text)
{
    //

    return {};//res;
}

std::wstring Mercury::Tokenizer::decode(std::vector<unsigned int> &localTokens)
{
    //

    return L"";//res;
}
