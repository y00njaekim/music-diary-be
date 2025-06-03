from langchain_openai import ChatOpenAI
import dotenv


dotenv.load_dotenv()
llm = ChatOpenAI(model="gpt-4.1", temperature=0)
