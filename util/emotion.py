class Emotion:
    def __init__(self):
        pass

    JOY = 0
    HOPE = 1
    NEUTRALITY = 2
    SADNESS = 3
    ANGER = 4
    ANXIETY = 5
    TIREDNESS = 6
    REGRET = 7

    def to_string(self, num):
        if num == self.JOY:
            return "HAPPY"
        if num == self.HOPE:
            return "HAPPY"
        if num == self.NEUTRALITY:
            return "NEUTRAL"
        if num == self.SADNESS:
            return "SAD"
        if num == self.ANGER:
            return "ANGRY"
        if num == self.ANXIETY:
            return "ANXIOUS"
        if num == self.TIREDNESS:
            return "TIRED"
        if num == self.REGRET:
            return "SAD"

    def to_num(self, st):
        st = st.strip()
        if st == "기쁨":
            return self.JOY
        if st == "희망":
            return self.HOPE
        if st == "중립":
            return self.NEUTRALITY
        if st == "슬픔":
            return self.SADNESS
        if st == "분노":
            return self.ANGER
        if st == "불안":
            return self.ANXIETY
        if st == "피곤":
            return self.TIREDNESS
        if st == "후회":
            return self.REGRET