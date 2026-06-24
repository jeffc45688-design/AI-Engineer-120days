from typing import Dict,List,Any
import time
from pydantic import BaseModel
from dataclasses import dataclass
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

class ModelResponse(BaseModel):
    success:bool
    status_code:int
    latency_ms:float
    data:Dict[str,Any]
    error:str|None = None

class QuestionRouter:
    def route(self,question:str) ->str:
        question_lower=question.lower()

        nutrition_keywords=["eat","diet","food","calorie","protein","fat loss","muscle gain"]
        workout_keywords=["train","workout","exercise","chest","legs","back","glutes"]

        if any(word in question_lower for word in nutrition_keywords):
            return "nutrition"
        if any(word in question_lower for word in workout_keywords):
            return "workout"
        
        return "general"
class NutritionTool:
    def run(self,question:str,user_profile:UserProfile) ->str:
        question_lower=question.lower()

        if "fat loss" in question_lower:
            return (
                f"For fat loss,focus on high protein,moderate carbs,"
                f"low fat,and calorie control."
                f"Your current goal is {user_profile.goal}"
            )
        if "muscle gain" in question_lower:
            return (
                f"For muscle gain,eat enough protein and stay in a calorie surplus."
                f"Your weight is {user_profile.weight}"
            )
        
        return "Eat balanced meals with protein ,carbs,healthy fats,and vegetables."
class WorkoutTool:
    def run(self,question:str,user_profile:UserProfile) ->str:
        question_lower=question.lower()

        if "chest" in question_lower:
            return "For chest training,use bench press,incline dumbbell press,and cable fly."
        if "legs" in question_lower or "glutes" in question_lower:
            return "For legs and glutes,use squats,hip thrusts,lunges,and Romanian deadlifts."
        
        return (
            f"Use progressive overload and choose training volume based on your level:"
            f"{user_profile.experience_level}"
        )
class SimpleRetriever:
    def __init__(self):
        self.documents=[
            "Fat loss requires a calorie deficit and enough protein",
            "Muscle gain requires calorie surplus,progressive overload and enough recovery.",
            "Chest training can include bench press,incline dumbbell press, and cable fly.",
            "Leg training can include squats,lunges,hip trusts,and Romanian deadlifts"
        ]
    def retrieve(self,question:str) ->List[str]:
        question_lower=question.lower().replace("?","")
        question_words=question_lower.split()
        results=[]
        for doc in self.documents:
            doc_lower=doc.lower()

            if any(word in doc_lower for word in question_words):
                results.append(doc)
        return results[:2]
class PromptBuilder:
    def build(self,question:str,tool_result:str,retrieved_docs:List[str],user_profile:UserProfile) ->str:
        context="\n".join(retrieved_docs)
        prompt=f"""
You are a helpful fitness AI assistant.

User Profile:
Goal:{user_profile.goal}
Weight:{user_profile.weight}
Experience Level:{user_profile.experience_level}

User Question:
{question}

Tool Result:
{tool_result}

Retrieved knowledge:
{context}

Give a clear and practical answer:
"""
        return prompt
class ServedModel:
    def __init__(self):
        self.model_name="local-fitness-ai-model"
        self.model_version="1.0.0"
        self.loaded=True
    def generate(self,prompt:str) ->str:
        return "This is the model-generated answer based on the prompt"

class FitnessAIApp:
    def __init__(self):
        self.router=QuestionRouter()
        self.nutrition_tool=NutritionTool()
        self.workout_tool=WorkoutTool()
        self.retriever=SimpleRetriever()
        self.prompt_builder=PromptBuilder()
        self.model=ServedModel()
    def answer(self,question:str,user_profile:UserProfile) ->Dict[str,Any]:
        route=self.router.route(question)

        if route == "nutrition":
            tool_result=self.nutrition_tool.run(question,user_profile)
        elif route == "workout":
            tool_result=self.workout_tool.run(question,user_profile)
        else:
            tool_result="This is a general fitness question"
        retrieved_docs=self.retriever.retrieve(question)
        prompt=self.prompt_builder.build(
            question=question,
            tool_result=tool_result,
            retrieved_docs=retrieved_docs,
            user_profile=user_profile
        )
        final_answer=self.model.generate(prompt)

        return {
            "question":question,
            "route":route,
            "tool_result":tool_result,
            "retrieved_docs":retrieved_docs,
            "prompt":prompt,
            "final_answer":final_answer,
            "model_name":self.model.model_name,
            "model_version":self.model.model_version
        }           
app=FastAPI(
    title="Day73 Serving AI Models",
    description="A simple model serving API for a fitness AI assistant.",
    version="1.0.0"
)
fitness_app=FitnessAIApp()
@app.get("/")
def home():
    return {"message":"Day73 Serving AI Models API is running."}

@app.get("/health")
def health_check():
    return {
        "status":"ok",
        "model_loaded":fitness_app.model.loaded,
        "model_name":fitness_app.model.model_name,
        "model_version":fitness_app.model.model_version
    }
@app.post("/predict",response_model=ModelResponse)
def predict(request:FitnessRequest):
    start_time=time.time()

    try:
        user_profile=UserProfile(
            goal=request.goal,
            weight=request.weight,
            experience_level=request.experience_level
        )
        result=fitness_app.answer(
            question=request.question,
            user_profile=user_profile
        )
        end_time=time.time()
        latency_ms=round((end_time-start_time)*1000,2)
        return ModelResponse(
            success=True,
            status_code=200,
            latency_ms=latency_ms,
            data=result,
            error=None
        )
    except Exception as e:
        end_time=time.time()
        latency_ms=round((end_time-start_time)*1000,2)
        return ModelResponse(
            success=False,
            status_code=500,
            latency_ms=latency_ms,
            data={},
            error=str(e)
        )