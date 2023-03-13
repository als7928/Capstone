# Capstone
Capstone Design in ICT, Spring 2023
---
### git 응용프로그램 사용 X. 터미널에서 명령어로만 하기
## 끝말잇기 해보기
### 튜토리얼
1. git clone https://github.com/als7928/Capstone.git
   - 적절한 위치에 clone 또는 pull 하세요
2. git branch 브랜치명
   - 자신의 이름으로 된 branch를 만드세요
3. git checkout 브랜치명
   - 자신의 branch로 이동하세요
4. txt파일 수정
   - txt 파일을 수정하세요
5. git add .
   - 변경 파일(.)을 main에 올릴 준비
6. git commit -m "~~~"
   - 내용적기
7. git push
   - minhyuk->minhyuk 최종 업데이트
   - 안되면 세팅해야함 git push --set-upstream origin 브랜치명
   - 원격저장소(깃허브) 본인 branch에서 변경사항 확인해보세요
8. git checkout main
   - main으로 오세요
9.  git merge 본인브랜치명
    - merge해서 본인이 수정한거 main에 넣으세요(잠시만 자동 PR 해놨음)
10. git push
    - minhyuk->main 최종 업데이트
    - 원격저장소(깃허브) main에서 변경사항 확인해보세요
11. 7번까지는 자신의 브랜치에서만 코드를 건드는 거라 main에 아무런 영향을 주지 않습니다
12. 8번 이후의 과정은 자동 Pull Request 설정을 해놨기 때문에 push하는 즉시 main에 반영됩니다 (원래는 여기 단계에서 관리자 허락 필요)
13. 코드 수정 전 항상 git pull 하는 습관을 들여야 동기화가 잘 됩니다
