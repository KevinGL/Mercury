#include "Mercury.h"

void Mercury::Network::Init(const unsigned int nbTokens)
{
    for(size_t i = 0 ; i < 2 * MERCURY_MAX_SIZE_EMBEDDINGS ; i++)
    {
        Neuron n;

        n.bias = (rand() % 101) / 100.0f;

        layers["input"].neurons.push_back(n);
    }

    const unsigned int nbLayersHidden = 2;//rand() % (5 - 2 + 1) + 2;

    for(size_t i = 0 ; i < nbLayersHidden ; i++)
    {
        std::ostringstream os;

        os << i + 1;

        unsigned int nbNeurons;

        const unsigned int nb = rand() % 3;

        /*if(nb == 0)
        {
            nbNeurons = 128;
        }

        else
        if(nb == 1)
        {
            nbNeurons = 256;
        }

        else
        if(nb == 2)
        {
            nbNeurons = 512;
        }*/

        nbNeurons = 16;

        for(size_t j = 0 ; j < nbNeurons ; j++)
        {
            Neuron n;

            n.bias = (rand() % 101) / 100.0f;

            layers["hidden" + os.str()].neurons.push_back(n);
        }
    }

    for(size_t i = 0 ; i < nbTokens ; i++)
    {
        Neuron n;

        n.bias = (rand() % 101) / 100.0f;

        layers["output"].neurons.push_back(n);
    }

    /*for(const auto& kv : layers)
    {
        std::cout << kv.first << " " << kv.second.neurons.size() << " neurons" << std::endl;
    }*/

    //////////////////////////////////////////
    //////////////////////////////////////////

    weights.resize(layers.size() - 1);

    weights[0].resize(layers["input"].neurons.size());

    for(size_t i = 0 ; i < layers["input"].neurons.size() ; i++)
    {
        weights[0][i].resize(layers["hidden1"].neurons.size());

        for(size_t j = 0 ; j < layers["hidden1"].neurons.size() ; j++)
        {
            /*std::ostringstream os1;
            os1 << i;

            std::ostringstream os2;
            os2 << j;

            const std::string key = "input#" + os1.str() + "_hidden1#" + os2.str();

            weights[key] = (rand() % 201 - 100) / 100.0f;*/

            //weights[0][i][j] = (rand() % 201 - 100) / 100.0f;
            weights[0][i][j] = (rand() % 101 - 50) / 100.0f;
        }
    }

    size_t index = 1;
    while(1)
    {
        std::ostringstream layer1;
        layer1 << index + 0;

        std::ostringstream layer2;
        layer2 << index + 1;

        const std::string nameLayer1 = "hidden" + layer1.str();
        const std::string nameLayer2 = "hidden" + layer2.str();

        if(layers.count(nameLayer2) == 0)
        {
            break;
        }

        weights[index].resize(layers[nameLayer1].neurons.size());

        for(size_t i = 0 ; i < layers[nameLayer1].neurons.size() ; i++)
        {
            weights[index][i].resize(layers[nameLayer2].neurons.size());

            for(size_t j = 0 ; j < layers[nameLayer2].neurons.size() ; j++)
            {
                /*std::ostringstream neuronLayer1;
                neuronLayer1 << i;

                std::ostringstream neuronLayer2;
                neuronLayer2 << j;

                const std::string key = "hidden" + layer1.str() + "#" + neuronLayer1.str() + "_hidden" + layer2.str() + "#" + neuronLayer2.str();

                //std::cout << key << std::endl;

                weights[key] = (rand() % 201 - 100) / 100.0f;*/

                //weights[index][i][j] = (rand() % 201 - 100) / 100.0f;
                weights[index][i][j] = (rand() % 101 - 150) / 100.0f;
            }
        }

        index++;
    }

    std::ostringstream nameLayerLast;
    nameLayerLast << index;

    indexLastLayerHidden = index;

    weights[indexLastLayerHidden].resize(layers["hidden" + nameLayerLast.str()].neurons.size());

    for(size_t i = 0 ; i < layers["hidden" + nameLayerLast.str()].neurons.size() ; i++)
    {
        weights[indexLastLayerHidden][i].resize(layers["output"].neurons.size());

        for(size_t j = 0 ; j < layers["output"].neurons.size() ; j++)
        {
            /*std::ostringstream os1;
            os1 << i;

            std::ostringstream os2;
            os2 << j;

            const std::string key = "hidden" + nameLayerLast.str() + "#" + os1.str() + "_output#" + os2.str();

            //std::cout << key << std::endl;

            weights[key] = (rand() % 201 - 100) / 100.0f;*/

            //weights[indexLastLayerHidden][i][j] = (rand() % 201 - 100) / 100.0f;
            weights[indexLastLayerHidden][i][j] = (rand() % 101 - 50) / 100.0f;
        }
    }

    std::cout << "Mercury : Network initialized" << std::endl;
}

void Mercury::Network::feedForward(std::vector<float> &input)
{
    if(!weights.size())
    {
        std::cout << "Mercury : Error, network not initialized" << std::endl;
        return;
    }

    for(size_t i = 0 ; i < input.size() ; i++)
    {
        if(i > layers["input"].neurons.size() - 1)
        {
            break;
        }

        layers["input"].neurons[i].value = input[i];
    }

    for(size_t i = 0 ; i < layers["hidden1"].neurons.size() ; i++)
    {
        std::ostringstream indexNeuronLayerActual;
        indexNeuronLayerActual << i;

        float sum = 0.0f;

        for(size_t j = 0 ; j < layers["input"].neurons.size() ; j++)
        {
            /*std::ostringstream indexNeuronLayerPrev;
            indexNeuronLayerPrev << j;

            const std::string key = "input#" + indexNeuronLayerPrev.str() + "_hidden1#" + indexNeuronLayerActual.str();

            //std::cout << key << " => " << weights.count(key) << std::endl;

            sum += weights[key] * layers["input"].neurons[j].value;*/

            sum += weights[0][j][i] * layers["input"].neurons[j].value;
        }

        layers["hidden1"].neurons[i].value = sum + layers["hidden1"].neurons[i].bias;
        layers["hidden1"].neurons[i].value = reLU(layers["hidden1"].neurons[i].value);
    }

    size_t index = 1;
    while(1)
    {
        std::ostringstream indexLayerPrev;
        indexLayerPrev << index + 0;

        std::ostringstream indexLayerActual;
        indexLayerActual << index + 1;

        const std::string nameLayerPrev = "hidden" + indexLayerPrev.str();
        const std::string nameLayerActual = "hidden" + indexLayerActual.str();

        if(layers.count(nameLayerActual) == 0)
        {
            break;
        }

        for(size_t i = 0 ; i < layers[nameLayerActual].neurons.size() ; i++)
        {
            std::ostringstream indexNeuronLayerActual;
            indexNeuronLayerActual << i;

            float sum = 0.0f;

            for(size_t j = 0 ; j < layers[nameLayerPrev].neurons.size() ; j++)
            {
                /*std::ostringstream indexNeuronLayerPrev;
                indexNeuronLayerPrev << j;

                const std::string key = nameLayerPrev + "#" + indexNeuronLayerPrev.str() + "_" + nameLayerActual + "#" + indexNeuronLayerActual.str();

                //std::cout << key << std::endl;

                sum += weights[key] * layers[nameLayerPrev].neurons[j].value;*/

                sum += weights[index][j][i] * layers[nameLayerPrev].neurons[j].value;
            }

            layers[nameLayerActual].neurons[i].value = sum + layers[nameLayerActual].neurons[i].bias;
            layers[nameLayerActual].neurons[i].value = reLU(layers[nameLayerActual].neurons[i].value);
        }

        index++;
    }

    std::ostringstream indexLayerPrev;
    indexLayerPrev << index;
    const std::string nameLayerPrev = "hidden" + indexLayerPrev.str();

    for(size_t i = 0 ; i < layers["output"].neurons.size() ; i++)
    {
        std::ostringstream indexNeuronLayerActual;
        indexNeuronLayerActual << i;

        float sum = 0.0f;

        for(size_t j = 0 ; j < layers[nameLayerPrev].neurons.size() ; j++)
        {
            /*std::ostringstream indexNeuronLayerPrev;
            indexNeuronLayerPrev << j;

            const std::string key = nameLayerPrev + "#" + indexNeuronLayerPrev.str() + "_output#" + indexNeuronLayerActual.str();

            //std::cout << key << std::endl;

            sum += weights[key] * layers[nameLayerPrev].neurons[j].value;*/

            sum += weights[index][j][i] * layers[nameLayerPrev].neurons[j].value;
        }

        layers["output"].neurons[i].value = sum + layers["output"].neurons[i].bias;
        //std::cout << layers["output"].neurons[i].value << std::endl;
    }

    //std::cout << "Mercury : Feed forward ok" << std::endl;

    //getch();
}

void Mercury::Network::backPropagation(std::vector<float> &vectorProba, std::vector<float> &vectorOneHot)
{
    if(!weights.size())
    {
        std::cout << "Mercury : Error, network not initialized" << std::endl;
        return;
    }

    /*for(auto& kv : layers)
    {
        for(Neuron &n : kv.second.neurons)
        {
            n.gradient = 0.0f;
        }
    }*/

    const float learningRate = 0.1f;

    std::string nameActualLayer = "output";
    std::string namePrevLayer = concatStringInt("hidden", indexLastLayerHidden);

    for(size_t i = 0 ; i < layers[nameActualLayer].neurons.size() ; i++)
    {
        const float gradient = vectorProba[i] - vectorOneHot[i];

        for(size_t j = 0 ; j < layers[namePrevLayer].neurons.size() ; j++)
        {
            //std::ostringstream os1, os2;

            //os1 << i;
            //os2 << j;

            //const std::string key = namePrevLayer + "#" + os2.str() + "_" + nameActualLayer + "#" + os1.str();

            //weights[key] -= learningRate * gradient * layers[namePrevLayer].neurons[j].value;

            weights[weights.size() - 1][j][i] -= learningRate * gradient * layers[namePrevLayer].neurons[j].value;
        }

        layers[nameActualLayer].neurons[i].bias -= learningRate * gradient;
        layers[nameActualLayer].neurons[i].gradient = gradient;
    }

    nameActualLayer = namePrevLayer;
    size_t indexPrevLayer = indexLastLayerHidden - 1;
    if(indexPrevLayer > 0)
    {
        namePrevLayer = concatStringInt("hidden", indexPrevLayer);
    }
    else
    {
        namePrevLayer = "input";
    }

    while(1)
    {
        for(size_t i = 0 ; i < layers[nameActualLayer].neurons.size() ; i++)
        {
            float gradient = 0.0f;

            for(size_t j = 0 ; j < layers[namePrevLayer].neurons.size() ; j++)
            {
                //std::ostringstream os1, os2;

                //os1 << i;
                //os2 << j;

                //const std::string key = nameActualLayer + "#" + os1.str() + "_" + namePrevLayer + "#" + os2.str();

                //gradient += layers[nameActualLayer].neurons[j].gradient * weights[key];

                float w = weights[indexPrevLayer][j][i];
                float g = layers[nameActualLayer].neurons[j].gradient;
                float prod = g * w;

                if (std::isnan(prod) || std::isinf(prod)) {
                    std::cout << "Anomaly detected at i=" << i << ", j=" << j << std::endl;
                    std::cout << "  weight = " << w << std::endl;
                    std::cout << "  gradient = " << g << std::endl;
                    std::cout << "  product = " << prod << std::endl;
                    std::cout << "  neuron value = " << layers[nameActualLayer].neurons[j].value << std::endl;
                    exit(1);
                }

                gradient += layers[nameActualLayer].neurons[j].gradient * weights[indexPrevLayer][j][i];
            }

            gradient *= derivReLU(layers[nameActualLayer].neurons[i].value);

            float clip = 1e3f;
            if (gradient > clip) gradient = clip;
            if (gradient < -clip) gradient = -clip;

            layers[nameActualLayer].neurons[i].gradient = gradient;

            for(size_t j = 0 ; j < layers[namePrevLayer].neurons.size() ; j++)
            {
                //std::ostringstream os1, os2;

                //os1 << i;
                //os2 << j;

                //const std::string key = namePrevLayer + "#" + os2.str() + "_" + nameActualLayer + "#" + os1.str();

                //weights[key] -= learningRate * gradient * layers[namePrevLayer].neurons[j].value;

                weights[indexPrevLayer][j][i] -= learningRate * gradient * layers[namePrevLayer].neurons[j].value;
                //std::cout << layers[namePrevLayer].neurons[j].value << " " << gradient << std::endl;
            }

            layers[nameActualLayer].neurons[i].bias -= learningRate * gradient;
        }

        //getch();

        if(namePrevLayer == "input")
        {
            break;
        }

        indexPrevLayer--;

        if(indexPrevLayer > 0)
        {
            namePrevLayer = concatStringInt("hidden", indexPrevLayer);
        }
        else
        {
            namePrevLayer = "input";
        }
    }

    //std::cout << "Mercury : Back propagation ok" << std::endl;
}

Mercury::Layer* Mercury::Network::getLayer(const std::string id)
{
    if(layers.count(id) == 1)
    {
        return &layers[id];
    }

    return nullptr;
}

void Mercury::Network::save(const std::string path)
{
    if(!weights.size())
    {
        std::cout << "Mercury : Error, network not initialized" << std::endl;
        return;
    }

    std::ofstream file(path);

    file << "----Layers----" << std::endl;

    for(const auto& layer : layers)
    {
        file << "    " << layer.first << std::endl;

        for(const Neuron &neuron : layer.second.neurons)
        {
            //file << "        " << neuron.bias << " " << neuron.value << std::endl;
            file << "        " << neuron.bias << std::endl;
        }
    }

    file << "----Weights----" << std::endl;

    /*for(const auto& weight : weights)
    {
        file << "    " << weight.first << " " << weight.second << std::endl;
    }*/

    for(const std::vector<std::vector<float>>& matrix : weights)
    {
        file << "                    ";

        for(size_t i = 0 ; i < matrix[0].size() ; i++)
        {
            std::ostringstream os;

            os << i;

            file << "n" << os.str();

            for(size_t j = os.str().length() + 1 ; j < 20 ; j++)
            {
                file << " ";
            }
        }

        file << std::endl;

        size_t index = 0;

        for(const std::vector<float>& line : matrix)
        {
            std::ostringstream os;

            os << index;

            file << "n" << os.str();

            for(size_t j = os.str().length() + 1 ; j < 20 ; j++)
            {
                file << " ";
            }

            for(const float weight : line)
            {
                file << weight;

                std::ostringstream os;

                os << weight;

                for(size_t j = os.str().length() ; j < 20 ; j++)
                {
                    file << " ";
                }
            }

            index++;

            file << std::endl;
        }

        file << std::endl;
    }

    file.close();
}

void Mercury::Network::loadDatas(const std::string path)
{
    std::ifstream file(path);

    if(!file)
    {
        return;
    }

    std::string datasToLoad = "";
    std::string nameLayer = "";
    std::string neuronDatas = "";

    size_t indexBetLayers = 0;
    size_t indexFromLayer = 0;
    size_t indexWeight = 0;

    std::vector<std::vector<float>> matrix;

    while(1)
    {
        std::string line;

        if(!getline(file, line))
        {
            break;
        }

        if(line.find("----") != std::string::npos)
        {
            datasToLoad = line;
            datasToLoad.erase(0, 4);
            datasToLoad.erase(datasToLoad.length() - 4);
        }

        else
        if(datasToLoad == "Layers")
        {
            if(nbRepeats(line, ' ') == 4)
            {
                nameLayer = line;
                nameLayer.erase(0, 4);
            }

            else
            {
                //if(nbRepeats(line, ' ') == 9)
                if(nbRepeats(line, ' ') == 8)
                {
                    neuronDatas = line;
                    neuronDatas.erase(0, 8);

                    Neuron neuron;
                    //sscanf(neuronDatas.c_str(), "%f %f\n", &neuron.bias, &neuron.value);
                    neuron.bias = atof(neuronDatas.c_str());

                    layers[nameLayer].neurons.push_back(neuron);
                }
            }
        }

        else
        if(datasToLoad == "Weights")
        {
            if(line.find("n") == 0)
            {
                std::string value = "";

                std::vector<float> lineFloat;

                for(size_t i = 20 ; i < line.length() ; i++)
                {
                    if(line[i] != ' ')
                    {
                        value += line[i];
                    }

                    else
                    {
                        if(value != "")
                        {
                            lineFloat.push_back(atof(value.c_str()));
                        }

                        value = "";
                    }
                }
                lineFloat.push_back(atof(value.c_str()));

                matrix.push_back(lineFloat);
            }

            else
            if(line == "")
            {
                weights.push_back(matrix);
                matrix.clear();
            }
        }
    }

    file.close();
}

void Mercury::Network::clear()
{
    layers.clear();
    weights.clear();
}
