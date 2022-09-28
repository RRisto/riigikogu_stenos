## Estonian Parliament (Riigikogu) stenograms topic analysis

This is just a hobby project for collecting stenograms from 
https://api.riigikogu.ee/swagger-ui.html and analyzing topics using simple NLP tools. 

My analysis pipeline looks following:
- `0_1_get_paevakorrad_stenod.ipynb` - collect stenograms
- `0_2_parse_data2df.ipynb` - parse data into pandas DataFrame
- `0_3_clean_text.ipynb` - do a bit of text cleaning
- `0_4_add_factions.ipynb` - collect faction info and merge 
 it stenograms dataframe
- `0_5_train_wordvectors.ipynb` - prepare data and train wav2vec and fasttext wordvectors (needed in step 10)
- `0.5_1_topic_modelling.ipynb` - topic modelling notebook you have to 
implement yourself ([LDA](https://radimrehurek.com/gensim/models/ldamodel.html) might be
 a good start).
I am using topic modelling solution from 
[Feelingstream](https://www.feelingstream.com/). 
- `0_6_analyze_topics.ipynb` - analyze overall topics, their yearly differences
- `0_7_analyze_parties_topics.ipynb` - analyze topics of parties, their differences
- `0_8_analyze_speakers.ipynb` - analyze some speakers and their differences
- `0_9_environment_topics_risto.ipynb` - analyze environment-related topics
- `10_0_gender_bias_w2v.ipynb` - calculate wav2vec (trained in  `0_5_train_wordvectors.ipynb`) word embeddings gender bias scores 
- `10_1_gender_bias_ft.ipynb` - calculate fasttext (trained in  `0_5_train_wordvectors.ipynb`) word embeddings gender bias scores 
- `10_2_analyze_gender_bias.ipynb` - analyze gender biases of wav2vec and fasttext embeddings 


## Usage
- Clone repo: `git clone https://github.com/RRisto/riigikogu_stenos.git`

Then build local environment or a docker container.

#### Local environment
- Install requirements from requirements.txt and run notebooks 
in your `jupyter-notebook` (or `jupyterhub`) server (need to install 
notebook server separately not included in requirements).
- Unpack data by running notebook: `0_0_unpack_existing_data.ipynb`


#### Docker 
It might be easier to build docker image and run docker container (includes `jupyter-notebook` server).

- Build and run docker:
<pre><code>"docker/build.bat" (or "docker/build.sh" for unix) 
"docker/run.bat" (or "docker/run.sh" for unix)</code></pre>
 On terminal you should see the following lines: 

 ![](https://github.com/RRisto/riigikogu_stenos/blob/master/images/server.PNG)

Click on the one that starts with `http://127.0.0.1:8888...` 
  
Now you should be inside folder with notebooks from this repo.

- Unpack data by running notebook: `0_0_unpack_existing_data.ipynb`
- Run other notebooks

## To repeat full analysis

If you run notebooks 0_1 - 0_4 yourself, you need to implement:
 
 - **topic modelling** solution which detects topics from text segments
 - solution to **detect changes** in topic proportions 
 (simplest would be to get difference of proportions between two groups)

I am using solutions from [Feelingstream](https://www.feelingstream.com/) 
for above mentioned two tasks and thus are not public.

If you just run notebooks from 0_5 - ... 
data is in data folder, you can rerun the analysis. Intermediate outputs (models, datafiles) are not included if they
exceed github maximum file size limit
