from flask import Flask, request, render_template
from queryBackend import get_query_tf_idf
from queryBackend import get_query_bm25
from queryBackend import get_query_w2v
from queryBackend import get_query_d2v
from queryBackend import my_preprocess


app = Flask(__name__)

def search(query, methodType):
    query = my_preprocess(str(query))
    search_result = None
    if methodType == 'tf-idf':
        search_result = get_query_tf_idf(query)
    elif methodType == 'bm25':
        search_result = get_query_bm25(query)
    elif methodType == 'w2v':
        search_result = get_query_w2v(query)
    elif methodType == 'd2v':
        search_result = get_query_d2v(query)
    else:
        return [("-1", f"error: method '{methodType}' does not exist")]
    return search_result
 
 
@app.route('/')
def index():
    if request.args:
        query = request.args['query']
        if len(query.strip()) == 0:
            return render_template('index.html',
                                    links=[("-1", f"error: query must be not empty")])
        methodType =request.args['methodType']
        links = search(query, methodType)
        return render_template('index.html', links=links)
    return render_template('index.html',links=[])

if __name__ == '__main__':
    app.run()
            
