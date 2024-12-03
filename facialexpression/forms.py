from django.forms import Form,CharField,FileField,PasswordInput

class LoginForm(Form):
    username = CharField(max_length=100)
    password = CharField(widget=PasswordInput())