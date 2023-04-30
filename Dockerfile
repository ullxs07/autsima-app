FROM python:3.9
RUN pip install back4app Flask tensorflow numpy pandas scikit-learn
WORKDIR /app
COPY app.py /app
COPY model.pkl /app
EXPOSE 5000
CMD ["python", "app.py"]
docker build -t my_ml_app.
docker run -p 5000:5000 my_ml_app
