# -*- coding: utf-8 -*-

"""
Created on Mon Jul 17 09:18:19 2017
@author: XuGang
"""
from __future__ import absolute_import
from game.agent import Agent

from flask import Flask
from flask import request
from datetime import timedelta
from flask import make_response, request, current_app, send_from_directory
from functools import update_wrapper

def crossdomain(origin=None, methods=None, headers=None,
                max_age=21600, attach_to_all=True,
                automatic_options=True):

    try:
        basestring
    except NameError:
        basestring = str
    
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, basestring):
        headers = ', '.join(x.upper() for x in headers)
    if not isinstance(origin, basestring):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        if methods is not None:
            return methods
        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        def wrapped_function(*args, **kwargs):
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp
            h = resp.headers
            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp
        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)
    return decorator

app = Flask(__name__, static_url_path='')

@app.route('/init',methods=['POST','GET'])
@crossdomain(origin='*')
def init():
    global agent
    model1 = request.form["model1"]
    model2 = request.form["model2"]
    model3 = request.form["model3"]
    
    agent.game_init(models=[model1, model2, model3], train=False)
    agent.game.get_next_moves()
    record = agent.game.get_record()
    
    return record

@app.route('/play',methods=['POST','GET'])
@crossdomain(origin='*')
def play():
    global agent
    try:
        if request.form["action_id"] == "":
            action_id = ""
        else:
            action_id = int(request.form["action_id"])
    except:
        action_id = None
    agent.next_move(action=action_id)
    record = agent.game.get_record()
    return record

@app.route('/ddz',methods=['GET'])
@crossdomain(origin='*')
def next_move():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    global agent
    agent = Agent()
    app.run(host='0.0.0.0', port=5000)
    

