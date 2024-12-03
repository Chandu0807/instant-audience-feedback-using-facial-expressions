from django.db import models

# Create your models here.
from django.db.models import Model


class FeedBackModel(Model):
    start_time=models.CharField(max_length=50)
    end_time=models.CharField(max_length=50)
    pcount=models.CharField(max_length=50)
    ncount=models.CharField(max_length=50)

class CurrentFeedback(Model):
    pcount = models.CharField(max_length=50)
    ncount = models.CharField(max_length=50)