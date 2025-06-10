#include "Mercury.h"

bool Mercury::isAlNum(const wchar_t ch)
{
    std::vector<wchar_t> specialChars = {L'â', L'ê', L'î', L'ô', L'û', L'é', L'à', L'è', L'ì', L'ò', L'ù', L'-', L'?', L'\''};
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
            if(isAlNum(corpusTrim[i + j]))
            {
                group += corpusTrim[i + j];
            }
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

float Mercury::reLU(const float value)
{
    return value >= 0.0f ? value : 0.0f;
}

float Mercury::derivReLU(const float value)
{
    return value >= 0.0f ? 1.0f : 0.0f;
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

void Mercury::softmax(Layer *layer, std::vector<float> &res)
{
    float maxVal = -1000.0f;
    std::vector<float> expVals;

    for(const auto& neuron : layer->neurons)
    {
        if(neuron.value > maxVal)
        {
            maxVal = neuron.value;
        }
    }

    float sum = 0.0f;

    for(const auto& neuron : layer->neurons)
    {
        float e = std::exp(neuron.value - maxVal);
        expVals.push_back(e);
        sum += e;
    }

    for(float e : expVals)
    {
        res.push_back(e / sum);
    }
}

size_t Mercury::getIndexMax(std::vector<float> &values)
{
    size_t res;
    float valMax = -1.0f;

    for(size_t i = 0 ; i < values.size() ; i++)
    {
        if(values[i] > valMax)
        {
            valMax = values[i];
            res = i;
        }
    }

    return res;
}

std::vector<float> Mercury::getVectorOneHot(const size_t index, const unsigned int nbTokens)
{
    std::vector<float> res;

    //for(size_t i = 0 ; i < MERCURY_MAX_TOKENS_OUTPUT_LAYER ; i++)
    for(size_t i = 0 ; i < nbTokens ; i++)
    {
        if(i != index)
        {
            res.push_back(0.0f);
        }

        else
        {
            res.push_back(1.0f);
        }
    }

    return res;
}

float Mercury::getCrossEntropy(std::vector<float> &vectorProba, std::vector<float> &vectorAttempted, const unsigned int nbTokens)
{
    float res = 0.0f;

    //for(size_t i = 0 ; i < MERCURY_MAX_TOKENS_OUTPUT_LAYER ; i++)
    for(size_t i = 0 ; i < nbTokens ; i++)
    {
        res -= vectorAttempted[i] * log(vectorProba[i]);
    }

    return res;
}

std::string Mercury::contactStringInt(std::string str, const unsigned int integer)
{
    std::ostringstream os;

    os << integer;

    return str + os.str();
}
