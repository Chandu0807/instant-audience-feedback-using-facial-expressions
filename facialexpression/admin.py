from django.contrib import admin

# Register your models here.
from facialexpression.models import FeedBackModel, CurrentFeedback

admin.site.register(FeedBackModel)
admin.site.register(CurrentFeedback)