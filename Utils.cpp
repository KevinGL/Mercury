#include "Mercury.h"

bool Mercury::isAlNum(const wchar_t ch)
{
    std::vector<wchar_t> specialChars = {L'â', L'ê', L'î', L'ô', L'û', L'é', L'à', L'è', L'ì', L'ò', L'ù', L' ', L'-'};
    std::vector<wchar_t> forbidden = {L',', L'.', L';', L':', L'|', L' '};

    if((std::isalnum(ch) || inArray(specialChars, ch)) && !inArray(forbidden, ch))
    {
        return true;
    }

    return false;
}

void Mercury::Tokenizer::getFirstTokens(const std::wstring corpus, unsigned int &id)
{
    tokens[L" "] = id;
    idToToken[id] = L" ";
    id++;

    for(const wchar_t ch : corpus)
    {
        if(isAlNum(ch))
        {
            std::wstring token = L"";
            token += ch;

            if(tokens.count(token) == 0)
            {
                tokens[token] = id;
                idToToken[id] = token;
                id++;
            }
        }
    }
}

std::map<std::wstring, unsigned int> Mercury::getGroupsFromCorpus(const std::wstring corpus, unsigned int groupsSize)
{
    if(groupsSize < 2)
    {
        return {};
    }

    std::map<std::wstring, unsigned int> groups;

    const std::wstring corpusTrim = trim(corpus);

    for(size_t i = 0 ; i < corpusTrim.length() - (groupsSize - 1) ; i++)
    {
        std::wstring group = L"";

        for(size_t j = 0 ; j < groupsSize ; j++)
        {
            group += corpusTrim[i + j];
        }

        groups[group]++;
    }

    return groups;
}

std::wstring Mercury::trim(const std::wstring text)
{
    std::wstring res = L"";

    for(const wchar_t ch : text)
    {
        if(ch != L' ' && ch != L'\t' && ch != '\n')
        {
            res += ch;
        }
    }

    return res;
}

unsigned int Mercury::wstringToInt(const std::wstring value)
{
    std::wstringstream wss(value);
    unsigned int res = 0;
    wss >> res;

    return res;
}

std::vector<std::wstring> Mercury::explode(std::wstring str, const wchar_t separator)
{
    std::vector<std::wstring> res;

    std::wstring item = L"";
    for(const wchar_t ch : str)
    {
        if(ch != separator)
        {
            item += ch;
        }

        else
        {
            res.push_back(item);
            item = L"";
        }
    }

    res.push_back(item);

    return res;
}

std::vector<std::wstring> Mercury::getMaxGroups(std::map<std::wstring, unsigned int> groups)
{
    /*unsigned int freqMax = 0;

    for(const auto &kv : groups)
    {
        if(kv.second > freqMax)
        {
            freqMax = kv.second;
        }
    }

    std::vector<std::wstring> res;

    for(const auto &kv : groups)
    {
        if(kv.second == freqMax)
        {
            res.push_back(kv.first);
        }
    }*/

    std::vector<std::wstring> res;

    for(const auto &kv : groups)
    {
        if(kv.second > 1)
        {
            res.push_back(kv.first);
        }
    }

    return res;
}
