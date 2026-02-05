from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import os

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'dev-secret-key-change-this'
    
    base_dir = os.path.abspath(os.path.dirname(__file__))
    
    # Use /tmp for database in Cloud Run to avoid read-only filesystem issues
    if os.environ.get('K_SERVICE') or os.environ.get('K_REVISION'):
        db_path = '/tmp/database.sqlite'
    else:
        db_path = os.path.join(os.path.dirname(base_dir), 'instance', 'database.sqlite')
        
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + db_path
    
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)

    from . import models

    with app.app_context():
        db.create_all()

    from .routes import main
    app.register_blueprint(main)

    return app
