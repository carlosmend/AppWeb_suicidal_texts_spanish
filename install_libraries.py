import subprocess

libs=['py -m pip install Flask', 'py -m pip install tensorflow','py -m pip install spacy','py -m pip install scikit-learn==1.2.2',
      'py -m pip install flask-cors','py -m spacy download es_core_news_lg']

for library in libs:
    subprocess.run(library, shell=True)
