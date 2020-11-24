# coco-nlp

Copy from Kashgari: https://github.com/BrikerMan/Kashgari/tree/v2-main/kashgari

Re-structured the code.

Add mongoDB generator.

All bert related layer is uploaded from bert4keras.

Based on tensorflow 2.2 version.

Deep Network structure is consisted by embedding layers (pre trained utilized feature extraction layer),
and feature extraction layers (extraction for specific data set).

Embedding layers are in coco_nlp/embeddings folder.

Feature extraction layers are in coco_nlp/tasks folder, it's classification, labeling and seq2seq separately.

This task support uploading train / test data from mongodb through mongoDB generator, 
and also local train / test data using corpus generator in coco_nlp/generators folder.


#### TODO
Add elasticsearch generator, DB generator.
Add more embedding layer that bert4keras can support in coco_nlp/embeddings folder.