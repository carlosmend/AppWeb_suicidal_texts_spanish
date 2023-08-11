import subprocess

libs=['pip install Flask', 'pip install tensorflow','pip install spacy','pip install scikit-learn==1.2.2'
'python -m spacy download es_core_news_lg']

for library in libs:
    subprocess.run(library, shell=True)
