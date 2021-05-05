web: python app.py
heroku ps:scale web=1
worker:bundle exec rake jobs:work
web: bundle exec rails server -p $PORT
web: gunicorn wsgi:app