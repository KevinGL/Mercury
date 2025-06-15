#include "Mercury.h"

void Mercury::Embedder::InitRandom(Mercury::Tokenizer &tokenizer)
{
    for(const auto& kv : tokenizer.getTokens())
    {
        std::vector<float> embedding;

        for(size_t i = 0 ; i < MERCURY_MAX_SIZE_EMBEDDINGS ; i++)
        {
            const float coord = (rand() % 201 - 100) / 100.0f;
            embedding.push_back(coord);
        }

        normalize(embedding);

        embeddings[kv.second] = embedding;
    }

    std::cout << "Embeddings init ok" << std::endl;
}

void Mercury::Embedder::assimilate(std::vector<std::wstring> &corpusText, Tokenizer &tokenizer)
{
    const float rate = 0.1f;

    for(const std::wstring line : corpusText)
    {
        std::vector<unsigned int> tokens = tokenizer.encode(line);

        if(tokens.size())
        {
            for(size_t i = 0 ; i < tokens.size() - 2 ; i++)
            {
                const unsigned int context1 = tokens[i + 0];
                const unsigned int subject = tokens[i + 1];
                const unsigned int context2 = tokens[i + 2];

                for(size_t j = 0 ; j < MERCURY_MAX_SIZE_EMBEDDINGS ; j++)
                {
                    if(embeddings.count(context1) && embeddings.count(subject) && subject != 1 && context1 != 1)
                    {
                        embeddings[context1][j] += rate * (embeddings[subject][j] - embeddings[context1][j]);
                        embeddings[subject][j] += rate * (embeddings[context1][j] - embeddings[subject][j]);
                    }

                    if(embeddings.count(context2) && embeddings.count(subject) && subject != 1 && context2 != 1)
                    {
                        embeddings[context2][j] += rate * (embeddings[subject][j] - embeddings[context2][j]);
                        embeddings[subject][j] += rate * (embeddings[context2][j] - embeddings[subject][j]);
                    }
                }

                normalize(embeddings[subject]);
                normalize(embeddings[context1]);
                normalize(embeddings[context2]);
            }
        }
    }
}

void Mercury::Embedder::learn(const std::string path, Tokenizer &tokenizer)
{
    embeddings.clear();
    predictionNetwork.clear();

    InitNetwork(tokenizer);
    //std::cout << tokenizer.getTokens().size() << " tokens " << predictionNetwork.getLayer("output")->neurons.size() << " neurones" << std::endl;

    std::vector<unsigned int> arrayIds = tokenizer.getArrayIds();

    InitRandom(tokenizer);

    //std::vector<std::pair<unsigned int, unsigned int>> pairs;
    std::vector<std::vector<unsigned int>> triplets;

    std::wifstream file(path + "/Mercury/Corpus2.txt");

    std::vector<std::wstring> corpusText;

    while(1)
    {
        std::wstring line;

        if(!getline(file, line))
        {
            break;
        }

        corpusText.push_back(line);
    }

    file.close();

    for(unsigned int pass = 0 ; pass < 10 ; pass++)
    {
        assimilate(corpusText, tokenizer);
    }

    std::ofstream file2(path + "/Mercury/Embeddings.txt");

    for(const auto& kv : embeddings)
    {
        file2 << kv.first << " [";

        for(size_t i = 0 ; i < kv.second.size() ; i++)
        {
            file2 << kv.second[i];

            if(i < kv.second.size() - 1)
            {
                file2 << " ";
            }
        }

        file2 << "]" << std::endl;
    }

    file2.close();

    /*for(const std::wstring line : corpusText)
    {
        std::vector<unsigned int> tokens = tokenizer.encode(line);

        if(tokens.size())
        {
            for(size_t i = 0 ; i < tokens.size() - 2 ; i++)
            {
                const unsigned int context1 = tokens[i + 0];
                const unsigned int subject = tokens[i + 1];
                const unsigned int context2 = tokens[i + 2];

                if(subject != 1 && context1 != 1)
                {
                    const float dot = dotProduct(embeddings[subject], embeddings[context1]);

                    std::wcout << tokenizer.getIds()[context1] << L" " << tokenizer.getIds()[subject] << L" => " << dot << std::endl;
                    getch();
                }

                if(subject != 1 && context2 != 1)
                {
                    const float dot = dotProduct(embeddings[subject], embeddings[context2]);

                    std::wcout << tokenizer.getIds()[context2] << L" " << tokenizer.getIds()[subject] << L" => " << dot << std::endl;
                    getch();
                }
            }
        }
    }

    return;*/

    for(const std::wstring line : corpusText)
    {
        std::vector<unsigned int> ids = tokenizer.encode(line);

        /*std::wcout << line << std::endl;

        for(const unsigned int id : ids)
        {
            std::cout << id << std::endl;
        }*/

        if(ids.size() > 0)
        {
            std::pair<unsigned int, unsigned int> pair;

            for(size_t i = 0 ; i < ids.size() - 2 ; i++)
            {
                /*std::cout << ids[i + 0] << " => " << embeddings[ids[i + 0]].size() << std::endl;
                std::cout << ids[i + 1] << " => " << embeddings[ids[i + 1]].size() << std::endl << std::endl;*/

                std::vector<unsigned int> triplet;

                triplet.push_back(ids[i + 0]);
                triplet.push_back(ids[i + 1]);
                triplet.push_back(ids[i + 2]);

                triplets.push_back(triplet);
            }
        }
    }

    size_t indexToken = 0;

    //for(const std::pair<unsigned int, unsigned int> pair : pairs)
    for(const std::vector<unsigned int> triplet : triplets)
    {
        /*for(const float value : kv.second)
        {
            Neuron neuron;

            neuron.output = value;

            predictionNetwork.input.neurons.push_back(neuron);
        }*/

        //std::cout << pair.first << " " << pair.second << std::endl;

        //if(pair.first != 1 && pair.second != 1)     //Case spaces
        if(triplet[0] != 1 && triplet[1] != 1 && triplet[2] != 1)     //Case spaces
        {
            float crossEntropy;
            size_t counter = 0;

            while(1)
            {
                //std::vector<float> embedding = embeddings[pair.first];
                std::vector<float> embedding1 = embeddings[triplet[0]];
                std::vector<float> embedding2 = embeddings[triplet[2]];

                std::vector<float> embedding;

                embedding.insert(embedding.end(), embedding1.begin(), embedding1.end());
                embedding.insert(embedding.end(), embedding2.begin(), embedding2.end());

                predictionNetwork.feedForward(embedding);
                normalize(embedding);

                std::vector<float> vectorProba;
                softmax(predictionNetwork.getLayer("output"), vectorProba);
                const size_t indexMax = getIndexMax(vectorProba);
                const unsigned int tokenPredicted = arrayIds[indexMax];

                //const int indexAttempted = indexArray(arrayIds, pair.second);
                const int indexAttempted = indexArray(arrayIds, triplet[1]);

                //if(indexAttempted > -1 && indexAttempted < MERCURY_MAX_TOKENS_OUTPUT_LAYER)
                if(indexAttempted > -1 && indexAttempted < tokenizer.getTokens().size())
                {
                    std::vector<float> vectorOneHot = getVectorOneHot(indexAttempted, tokenizer.getTokens().size());
                    crossEntropy = getCrossEntropy(vectorProba, vectorOneHot, tokenizer.getTokens().size());

                    //std::cout << "Cross entropy : " << crossEntropy << std::endl;

                    if(std::isnan(crossEntropy) || std::isinf(crossEntropy))
                    {
                        std::cout << "Cross entropy value NaN or infinite, stop processus" << std::endl;

                        /*for(const float p : vectorProba)
                        {
                            std::cout << p << std::endl;
                        }*/

                        break;
                    }

                    //getch();

                    predictionNetwork.backPropagation(vectorProba, vectorOneHot);
                }

                if(crossEntropy < 0.05f)
                {
                    break;
                }

                counter++;
                if(counter > 100)
                {
                    break;
                }
            }

            /*std::wcout << L"Learning ok for " << tokenizer.getIds()[pair.first] << L" and " << tokenizer.getIds()[pair.second] << std::endl;
            getch();*/

            const std::wstring token1 = tokenizer.getIds()[triplet[0]];
            const std::wstring token2 = tokenizer.getIds()[triplet[2]];
            const std::wstring tokenPredicted = tokenizer.getIds()[triplet[1]];

            /*std::wcout << token1 << L" " << token2 << L") => " << tokenPredicted << std::endl;
            getch();*/
        }

        indexToken++;

        //std::cout << indexToken << " on " << pairs.size() << std::endl;
        std::cout << indexToken << " on " << triplet.size() << std::endl;
    }

    predictionNetwork.save(path + "/Mercury/PredictionNetwork.txt");
}

void Mercury::Embedder::InitNetwork(Mercury::Tokenizer &tokenizer)
{
    predictionNetwork.Init(tokenizer.getTokens().size());
}

void Mercury::Embedder::loadDatas(const std::string path)
{
    std::ifstream file(path + "/Mercury/Embeddings.txt");
    if(!file)
    {
        return;
    }

    while(1)
    {
        std::string line;

        if(!getline(file, line))
        {
            break;
        }

        std::vector<float> embedding;

        std::string keyStr = line;
        keyStr.erase(keyStr.find(" "));
        const unsigned int key = atoi(keyStr.c_str());

        std::string embeddingStr = line;
        embeddingStr.erase(0, embeddingStr.find(" ") + 1);

        std::string value = "";
        for(size_t i = 1 ; i < embeddingStr.length() - 1 ; i++)
        {
            if(embeddingStr.at(i) != ' ')
            {
                value += embeddingStr.at(i);
            }

            else
            {
                embedding.push_back(atof(value.c_str()));
                value = "";
            }
        }

        embedding.push_back(atof(value.c_str()));

        embeddings[key] = embedding;
    }

    file.close();

    predictionNetwork.loadDatas(path + "/Mercury/PredictionNetwork.txt");
}
