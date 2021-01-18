# Twitch game predictor

This project attempts to build an AI that can predict what game a streamer is playing just by analysing a single frame of the stream. I had this idea after forgetting to update my stream information correctly for too long and thinking about a way to fix this problem for good. As always, instead of going for a easy and boring route, I decided to revisit my computer vision projects and attempt to reapply everything I learned, this time using Python.

### Predictor v1

The very first version of this project was an implementation of the Bag of Words algorithm using the SURF extractor. I had lots of experience using this technique to classify microscopic pictures of pollen using machine learning classifiers (such as SMO and SVM).

Since I had a good result with pollen, I thought it would be a straight forward job to use in games since: 
1. I could use higher quality images; 
2. Most games have repeating image features (UI);
3. Games have a very distinct art style.

##### The dataset

For some reason, I decided to manually capture the pictures instead of automatically downloading  stream thumbnails. For this reason, this dataset had 3 versions:

1. 5 games with 10 images each;
2. 5 games with 20 images each;
3. 15 games with 30 images each.

##### Feature extraction

I had the option to use SIFT, SURF and ORB. I went for SURF since I knew exactly how I would implement the algorithm and had no experience with SIFT or ORB.

Since I was not planning on writing the results of each experiment, I randomly played around with the hessian threshold value, and decided that 250 got me the best results. Visual analysis confirmed that a lower threshold yielded more UI features.

##### Clustering

The entire project used k-means. As with hessian threshold, I decided to play around with the `k` parameter and decided on multiples of 1024. With 15 games and hessian threshold 250, I increased this parameter to 5 * 1024.

As expected, higher clusters, yielded VERY long clustering times. This was solved by using a CUDA implementation of k-means.

While clustering time was for the most part, solved, by using CUDA, resource usage stayed high. At this moment I figure that this technique would only work for a few games at best. I'll talk about this in the results section.

#### Classifiers

I decided to throw almost every classifier `sklearn` could offer me:

- KNN
- SVC
- GaussianNB
- QuadraticDiscriminantAnalysis
- DecisionTreeClassifier
- LogisticRegression
- MLPClassifier
- GaussianProcessClassifier

After a few results, I decided to only continue with MLP, LogisticRegression and SVC as they were the only classifiers achieving 60% or more accuracy.

##### Results

I don't have any raw numbers or graphs at this moment of writing as Google Colab is not letting me connect to their GPU runtimes because of usage limits.

The bottom line of this method is that training time grew at a rate that it wouldn't be possible to increase the amount of games and keep a decent classification accuracy (I'm aware that I don't need a high accuracy to solve this problem, since I'm able to load multiple stream frames, classify them and select the game with most occurrences).

The nature of this problem mean that for each game I added, I would have to increase my `k` value (because a different game with different art style would introduce more visual words in the mix), together with `n` more images to compute histograms of, meant that traning time would grow very fast.

The biggest problem was the feature pool size: each image had on average 5000 features (2.5 million in total with 480 images), meaning that increasing the dataset, in any way, would easily reach a point of needing special hardware (which is simply not worth it).

At around 80% accuracy, I decided to give up on this method and attempt something else.


### Predictor v2

After my first version not having a lot of room to grow, I decided to attempt a completely different and new (to me) method: convolutional neural networks. I've read about them, even studied how they work (thanks to my Computer Engineering degree) but never actually implemented one myself.

Almost everything in this version came from [this amazing TensorFlow tutorial on image classification](https://www.tensorflow.org/tutorials/images/classification).

##### Dataset for v2

I knew that 30 images would not cut for CNNs, so I decided to finally implement something to quickly download stream thumbnails from Twitch. I don't have the code public for it yet, but someday...

I decided on (around) 1000 images on the top 16 games at the momen: Apex Legends, CS:GO, Dead by Daylight, Escape from Tarkov, FIFA 21, Fortnite, GTA 5, League of Legends, Minecraft, Path of Exile, Rocket League, Rust, Valorant, Warzone, World of Warcraft.

##### CNN

Since I had (and still have) absolutely no idea how to go about parameters and structure, I just threw random numbers and checked for validation/classification loss.

I had multiple attempts running locally and on Google Colab to see if something worked.

Multiple attempts later, crossing 80% accuracy seemed to be impossible.

![](https://imgur.com/ZK7H3xd.png)

Since my dataset was completely untouched after downloading them from Twitch servers, I decided to manually check 16,000 images and check for bad images (no game, low bitrate, wrong game, too much overlay, etc).

This resulted in around 2,000 images being deleted.

While this improved the results, I still wasn't happy about it.

![](https://imgur.com/ziptkO8.png)

At this point, multiple parameters changes, structure changes, 85% seemed to be the limit.


##### Current situation

This problem got pretty interesting, since I didn't expect it to be difficult to get more than 90% accuracy.

I'll attempt to better document the changes I make in this README in an attempt to reach 95%+ with 15 games or around 90%+ with 50 (or more?) games.

I'm still not sure what my plans are with this project. But at this point I might only add games I would play myself so at the very least I can use this project in my streams.

### Things to experiment:

- [ ] Drastically increase dataset set (5k to 20k images);
- [ ] Use known CNN structures;
- [ ] Learn what I should be looking for in tweaking the CNN;
- [ ] Increase games from 15 to 30 and then 60;
- [ ] Input size influence;
- [ ] Actually read on what I should be doing instead of blindly attempting things.