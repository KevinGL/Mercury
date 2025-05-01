#include "Mercury.h"

bool Mercury::inArrayWchar(const std::vector<wchar_t> array, const wchar_t value)
{
    for(const wchar_t el : array)
    {
        if(el == value)
        {
            return true;
        }
    }

    return false;
}

bool Mercury::inArrayWstring(const std::vector<std::wstring> array, const std::wstring value)
{
    for(const std::wstring el : array)
    {
        if(el == value)
        {
            return true;
        }
    }

    return false;
}

bool Mercury::isAlNum(const wchar_t ch)
{
    std::vector<wchar_t> specialChars = {L'â', L'ê', L'î', L'ô', L'û', L'é', L'à', L'è', L'ì', L'ò', L'ù', L' ', L'-'};
    std::vector<wchar_t> forbidden = {L',', L'.', L';', L':'};

    if((std::isalnum(ch) || inArrayWchar(specialChars, ch)) && !inArrayWchar(forbidden, ch))
    {
        return true;
    }

    return false;
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

std::vector<std::wstring> Mercury::getMaxPairs(std::map<std::wstring, unsigned int> pairs)
{
    unsigned int freqMax = 0;

    for(const auto &kv : pairs)
    {
        if(kv.second > freqMax)
        {
            freqMax = kv.second;
        }
    }

    std::vector<std::wstring> res;

    for(const auto &kv : pairs)
    {
        if(kv.second == freqMax)
        {
            res.push_back(kv.first);
        }
    }

    return res;
}
