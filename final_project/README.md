# Final project

A simple search engine for the covid-19 dataset.
You need to download the fasttext model "araneum_none_fasttextcbow_300_5_2018.model" from https://rusvectores.org/ru/models/.

Unzip the following 4 files in the flask folder with the server:
araneum_none_fasttextcbow_300_5_2018.model.vectors_vocab.npy
araneum_none_fasttextcbow_300_5_2018.model
araneum_none_fasttextcbow_300_5_2018.model.vectors_ngrams.npy
araneum_none_fasttextcbow_300_5_2018.model.vectors.npy

For faster query processing, firstly create the necessary matrices for the methods with the command:
python3 create_matrices.py

After this you can run the server with the command:
python3 app.py
