#include "Mercury.h"

void Mercury::Network::Init(const unsigned int nbTokens)
{
    for(size_t i = 0 ; i < MERCURY_MAX_SIZE_EMBEDDINGS ; i++)
    {
        Neuron n;

        n.bias = (rand() % 101) / 100.0f;

        layers["input"].neurons.push_back(n);
    }

    const unsigned int nbLayersHidden = rand() % (5 - 2 + 1) + 2;

    for(size_t i = 0 ; i < nbLayersHidden ; i++)
    {
        std::ostringstream os;

        os << i + 1;

        unsigned int nbNeurons;

        const unsigned int nb = rand() % 3;

        if(nb == 0)
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
        }

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

    for(size_t i = 0 ; i < layers["input"].neurons.size() ; i++)
    {
        for(size_t j = 0 ; j < layers["hidden1"].neurons.size() ; j++)
        {
            std::ostringstream os1;
            os1 << i;

            std::ostringstream os2;
            os2 << j;

            const std::string key = "input#" + os1.str() + "_hidden1#" + os2.str();

            //weights[key] = (rand() % 101) / 100.0f;
            weights[key] = ((rand() / (float)RAND_MAX) - 0.5f) * sqrtf(2.0f / layers["input"].neurons.size());
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

        for(size_t i = 0 ; i < layers[nameLayer1].neurons.size() ; i++)
        {
            for(size_t j = 0 ; j < layers[nameLayer2].neurons.size() ; j++)
            {
                std::ostringstream neuronLayer1;
                neuronLayer1 << i;

                std::ostringstream neuronLayer2;
                neuronLayer2 << j;

                const std::string key = "hidden" + layer1.str() + "#" + neuronLayer1.str() + "_hidden" + layer2.str() + "#" + neuronLayer2.str();

                //std::cout << key << std::endl;

                //weights[key] = (rand() % 101) / 100.0f;
                weights[key] = ((rand() / (float)RAND_MAX) - 0.5f) * sqrtf(2.0f / layers[nameLayer1].neurons.size());
            }
        }

        index++;
    }

    std::ostringstream nameLayerLast;
    nameLayerLast << index;

    for(size_t i = 0 ; i < layers["hidden" + nameLayerLast.str()].neurons.size() ; i++)
    {
        for(size_t j = 0 ; j < layers["output"].neurons.size() ; j++)
        {
            std::ostringstream os1;
            os1 << i;

            std::ostringstream os2;
            os2 << j;

            const std::string key = "hidden" + nameLayerLast.str() + "#" + os1.str() + "_output#" + os2.str();

            //std::cout << key << std::endl;

            //weights[key] = (rand() % 101) / 100.0f;
            weights[key] = ((rand() / (float)RAND_MAX) - 0.5f) * sqrtf(2.0f / layers["hidden" + nameLayerLast.str()].neurons.size());
        }
    }

    std::cout << "Mercury : Network initialized" << std::endl;
}

void Mercury::Network::feedForward(std::vector<float> &input)
{
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
            std::ostringstream indexNeuronLayerPrev;
            indexNeuronLayerPrev << j;

            const std::string key = "input#" + indexNeuronLayerPrev.str() + "_hidden1#" + indexNeuronLayerActual.str();

            //std::cout << key << " => " << weights.count(key) << std::endl;

            sum += weights[key] * layers["input"].neurons[j].value;
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
                std::ostringstream indexNeuronLayerPrev;
                indexNeuronLayerPrev << j;

                const std::string key = nameLayerPrev + "#" + indexNeuronLayerPrev.str() + "_" + nameLayerActual + "#" + indexNeuronLayerActual.str();

                //std::cout << key << std::endl;

                sum += weights[key] * layers[nameLayerPrev].neurons[j].value;
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
            std::ostringstream indexNeuronLayerPrev;
            indexNeuronLayerPrev << j;

            const std::string key = nameLayerPrev + "#" + indexNeuronLayerPrev.str() + "_output#" + indexNeuronLayerActual.str();

            //std::cout << key << std::endl;

            sum += weights[key] * layers[nameLayerPrev].neurons[j].value;
        }

        layers["output"].neurons[i].value = sum + layers["output"].neurons[i].bias;
    }
}

Mercury::Layer* Mercury::Network::getLayer(const std::string id)
{
    if(layers.count(id) == 1)
    {
        return &layers[id];
    }

    return nullptr;
}
