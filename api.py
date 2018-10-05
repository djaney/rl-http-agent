#!/usr/bin/env python3
import os
from core.flask import FlaskApp
from core.model_factory import get_agent, WEIGHTS

dir_path = os.path.dirname(os.path.realpath(__file__))
weights_path = "{}/{}".format(dir_path, WEIGHTS)
agent = get_agent()

flask_app = FlaskApp(agent, weights_path)
flask_app.run(debug=True, host='0.0.0.0')