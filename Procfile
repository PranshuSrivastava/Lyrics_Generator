heroku ps:scale web=1
worker:bundle exec rake jobs:work
web: bundle exec rails server -p $PORT
web: gunicorn --bind 0.0.0.0:$PORT flaskapp:app