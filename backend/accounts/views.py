"""회원가입 화면. 로그인·로그아웃은 Django 기본 뷰(LoginView·LogoutView)를 연결해 사용한다."""

from django.contrib.auth import login
from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import redirect
from django.urls import reverse_lazy
from django.views.generic import CreateView


class SignUpView(CreateView):
    form_class = UserCreationForm
    template_name = "accounts/signup.html"
    success_url = reverse_lazy("analyses:index")

    def dispatch(self, request, *args, **kwargs):
        if request.user.is_authenticated:            # 이미 로그인했다면 가입 화면 대신 분석 화면으로
            return redirect(self.success_url)
        return super().dispatch(request, *args, **kwargs)

    def form_valid(self, form):
        response = super().form_valid(form)
        login(self.request, self.object)             # 가입 직후 자동 로그인
        return response
