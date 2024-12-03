import datetime

from django.http import StreamingHttpResponse
from django.shortcuts import render

from facialexpression.camera import *
from facialexpression.forms import LoginForm
from facialexpression.models import CurrentFeedback

def login(request):

    if request.method == "GET":
        # Get the posted form
        loginForm = LoginForm(request.GET)

        if loginForm.is_valid():

            uname = loginForm.cleaned_data["username"]
            upass = loginForm.cleaned_data["password"]

            if uname == "admin" and upass == "admin":
                request.session['role'] = "admin"
                return render(request, 'home.html', {"feedbacks":FeedBackModel.objects.all()})
            else:
                return render(request, 'index.html', {"message": "Invalid Credentials"})

        return render(request, 'index.html', {"message": "Invalid From"})

    return render(request, 'index.html', {"message": "Invalid Request"})

def logout(request):
    try:
        del request.session['username']
    except:
        pass
    return render(request, 'index.html', {})

def home(request):
    return render(request, 'home.html', {"feedbacks": FeedBackModel.objects.all()})


cam = VideoCamera()
def livefe(request):
    try:
        request.session["start_time"]=str(datetime.datetime.now())
        print(str(datetime.datetime.now()))
        CurrentFeedback(pcount=0,ncount=0).save()
        cam.start()
        result=gen(cam,request)
        return StreamingHttpResponse(result, content_type="multipart/x-mixed-replace;boundary=frame")
    except Exception as e:  # This is bad!
        print(e)
        pass

def stop(request):
    cam.stop()
    request.session["stop_time"] = str(datetime.datetime.now())
    cf=CurrentFeedback.objects.all().first()
    FeedBackModel(start_time=request.session["start_time"], end_time=request.session["stop_time"],
                  pcount=cf.pcount, ncount=cf.ncount).save()
    return render(request, 'result.html', {"pcount":cf.pcount,"ncount":cf.ncount})
