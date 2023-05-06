from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from agents.linkedin_lookup_agent import linkedin_lookup_agent
from agents.twitter_lookup_agent import twitter_lookup_agent
from third_parties.linkedin import scrape_linkedin_profile
from third_parties.twitter import scrape_user_tweets

information = """
Elon Reeve Musk FRS (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a business magnate and investor. He is the founder, CEO and chief engineer of SpaceX; angel investor, CEO and product architect of Tesla, Inc.; owner and CEO of Twitter; founder of the Boring Company; co-founder of Neuralink and OpenAI; and president of the philanthropic Musk Foundation. With an estimated net worth of around $192 billion as of March 27, 2023, primarily from his ownership stakes in Tesla and SpaceX,[4][5] Musk is the second-wealthiest person in the world, according to both the Bloomberg Billionaires Index and Forbes's real-time billionaires list.[6][7]
"""

if __name__ == "__main__":
    # linkedin_data = scrape_linkedin_profile(linkedin_profile_url="https://www.linkedin.com/in/harrison-chase-961287118")
    # print(linkedin_data)

    linkedin_profile_url = linkedin_lookup_agent(name="Ramesh")
    linkedin_data = scrape_linkedin_profile(
        # linkedin_profile_url="https://www.linkedin.com/in/ramesh-yenugula-61ba8734"
        linkedin_profile_url=linkedin_profile_url
    )

    twitter_username = twitter_lookup_agent(name="@elonmusk")
    tweets = scrape_user_tweets(username=twitter_username, run_tweets=5)

    summary_template = """
            Given the linkedin information {linkedin_information} and twitter {twitter_information} about a person from I want you to create:
            1. A short summary
            2. Two intresting facts about them
        """

    summary_prompt_template = PromptTemplate(
        input_variables=["linkedin_information", "twitter_information"],
        template=summary_template,
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    print(chain.run(linkedin_information=linkedin_data, twitter_information=twitter))

    scrape_user_tweets(username="@elonmusk")
