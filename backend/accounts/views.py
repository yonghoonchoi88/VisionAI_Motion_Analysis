"""회원가입·본인 정보 화면. 로그인·로그아웃·비밀번호 변경은 Django 기본 인증 뷰를 연결해 사용한다."""

from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.db.models import Count, Q
from django.shortcuts import redirect, render
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


@login_required
def profile(request):
    """본인 정보: 계정 정보 + 내 분석 요약 (다른 회원의 정보는 볼 수 없음)."""
    from analyses.constants import ACTION_ID, ACTION_KO
    from analyses.models import AnalysisJob, PersonPrediction

    jobs = AnalysisJob.objects.filter(user=request.user).order_by()
    stats = jobs.aggregate(
        total=Count("id"),
        success=Count("id", filter=Q(status=AnalysisJob.Status.SUCCESS)),
        failure=Count("id", filter=Q(status=AnalysisJob.Status.FAILURE)),
        running=Count("id", filter=Q(status__in=[AnalysisJob.Status.PENDING, AnalysisJob.Status.STARTED])),
        upload=Count("id", filter=Q(input_type=AnalysisJob.InputType.UPLOAD)),
        camera=Count("id", filter=Q(input_type=AnalysisJob.InputType.CAMERA)),
    )
    top_actions = (
        PersonPrediction.objects.filter(job__user=request.user).order_by()
        .values("action_label").annotate(n=Count("id")).order_by("-n")[:3]
    )
    stats["predictions"] = PersonPrediction.objects.filter(job__user=request.user).count()
    top = [
        {"label": r["action_label"], "action_id": ACTION_ID.get(r["action_label"], ""),
         "ko": ACTION_KO.get(r["action_label"], ""), "n": r["n"]}
        for r in top_actions
    ]
    return render(request, "accounts/profile.html", {"stats": stats, "top_actions": top})
