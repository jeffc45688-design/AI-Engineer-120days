from typing import Dict,Any,List
from dataclasses import dataclass 
from pydantic import BaseModel
from fastapi import FastAPI

@dataclass
class UserProfile:
    goal:str
    weight:int
    experience_level:str

class FitnessRequest(BaseModel):
    question:str
    goal:str
    weight:int
    experience_level:str

class QuestionRouter:
    def route(self,question:str) -> str:
        question_lower=question.lower()

        nutrition_keywards=["eat","diet","food","calorie","protein","fat loss","muscle gain"]
        workout_keywards=["train","workout","exercise","chest","legs","back","glutes"]

        if any(word in question_lower for word in nutrition_keywards):
            return "nutrition"
        if any(word in question_lower for word in workout_keywards):
            return "workout"
        
        return "general"
class NutritionTool:
    def run(self,question:str,user_profile:UserProfile) -> str:
        question_lower=question.lower()

        if "fat loss" in question_lower:
            return (
                f"For fat loss,focus on high protein,moderate carbs, "
                f"low fat,and calorie comtrol. "
                f"Your current goal is {user_profile.goal}"
            )
        if "mucle gain" in question_lower:
            return (
                f"For muscle gain,eat enough protein and stay in a calorie surplus."
                f"Your weight is {user_profile.weight}"
            )
        return "Eat balanced meals with protein ,carbs,healthy fats,and vegetables."
class WorkoutTool:
    def run(self,question:str,user_profile:UserProfile) -> str:
        question_lower=question.lower()

        if "chset" in question_lower:
            return "For chest training,use bench press,incline dumbbell press,and cable fly."
        if "legs" in question_lower or "glutes" in question_lower:
            return "For legs and glutes,use squats,hip trusts,lunges,and Romanian deadlifts."
        
        return (
            f"Use progressive overload and choose training volume based on your experience level."
            f"Your level is {user_profile.experience_level}"
        )
class SimpleRetriever:
    def __init__(self):
        self.ducuments=[
            "Fat loss requires a calories deficit and enough protein.",
            "Muscle gain requires calories surplus,progressive overload and enough recovery.",
            "Chest training can include bench press,incline dumbbell press, and cable fly.",
            "Leg training can include squats,lunges,hip trusts,and Romanian deadlifts."
        ]
    def retrieve(self,question:str) ->List[str]:
        question_lower=question.lower().replace("?","") 
        results=[]
        for doc in self.ducuments:
            doc_lower=doc.lower()
            
            if any(word in doc_lower for word in question_lower.split()):
                results.append(doc)

        return results[:2]
class PromptBuilder:
    def build(self,question:str,tool_result:str,retrieved_docs:List[str],user_profile:UserProfile) -> str:
        context="/".join(retrieved_docs)
        prompt=f"""
You are a helpful Fitness AI assistant.
UserProfile:
Goal:{user_profile.goal}
Weight:{user_profile.weight}
Experience Level:{user_profile.experience_level}

Question:
{question}

Tool Result:
{tool_result}

Retrieved knowledge:
{context}

Give a clear and practical answer: 
"""
        return prompt
class LocalLLM:
    def generate(self,prompt:str) ->str:
        return "This is where the LLM would generate the final answer based on the prompt"

class FitnessAIApp:
    def __init__(self):
        self.router=QuestionRouter()
        self.nutrition_tool=NutritionTool()
        self.workout_tool=WorkoutTool()
        self.retriever=SimpleRetriever()
        self.prompt_builder=PromptBuilder()
        self.llm=LocalLLM()
    def answer(self,question:str,user_profile:UserProfile) -> Dict[str,Any]:
        route=self.router.route(question)

        if route == "nutrition":
            tool_result=self.nutrition_tool.run(question,user_profile)

        elif route == "workout":
            tool_result=self.workout_tool.run(question,user_profile)

        else:
            tool_result="This is a general fitness question."

        retrieved_docs=self.retriever.retrieve(question)

        prompt=self.prompt_builder.build(
            question=question,
            tool_result=tool_result,
            retrieved_docs=retrieved_docs,
            user_profile=user_profile,
        )
        final_answer=self.llm.generate(prompt)

        return {
            "question":question,
            "route":route,
            "tool_result":tool_result,
            "retrieved_docs":retrieved_docs,
            "prompt":prompt,
            "final_answer":final_answer
        }

app=FastAPI()
fitness_app=FitnessAIApp()

@app.get("/")
def home():
    return {
        "message":"Day72 FastAPI Basics is runnung."
    }

@app.get("/health")
def health_check():
    return {
        "status":"ok",
        "service":"fitness_ai_api"
    }

@app.post("/answer")
def answer_question(request:FitnessRequest):
    user_profile=UserProfile(
        goal=request.goal,
        weight=request.weight,
        experience_level=request.experience_level
    )
    result=fitness_app.answer(
        question=request.question,
        user_profile=user_profile
    )
    return {
        "success":True,
        "status_code":200,
        "error":None,
        "data":result
    }
