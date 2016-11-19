# Chess ratings

This code uses an SVM to predict the strength of chess players. There
are three experiments implemented: classification, ranking, and regression.

You will need the following Python packages:

* NLTK
* sklearn
* pgn

The chess commentary data is available from here:

http://www.ims.uni-stuttgart.de/data/chess

To start the experiment simply run:

`python svm.py`

The results of this experiment are described in this paper:

    @InProceedings{sentimentrelevance,
        author =       {Christian Scheible and Hinrich Sch\"utze},
        title =        {Picking the Amateur's Mind - Predicting Chess Player Strength from Game Annotations },
        booktitle =    "Proceedings of Coling 2014",
        year =         2014
    }

