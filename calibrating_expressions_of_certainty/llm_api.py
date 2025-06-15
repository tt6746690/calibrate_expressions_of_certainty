import time, os
from openai import RateLimitError as OpenAIRateLimitError
from openai import OpenAI
import anthropic
from anthropic import RateLimitError as AnthropicRateLimitError


openai_client = OpenAI()
anthropic_client = anthropic.Anthropic()

def get_completion_response_openai(
    prompt,
    model='gpt-4o-mini',
    temperature=0,
    seed=None,
    max_tokens=1024,
    max_retries=10_000,
    retry_delay=5,
):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt,}
    ]
    for attempt in range(max_retries):
        try:
            completion = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                seed=seed,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content
        except OpenAIRateLimitError:
            if attempt < max_retries - 1:  # If it's not the last attempt
                print(f"Rate limit reached. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise  # Re-raise the exception if we've exhausted all retries


def get_completion_response_anthropic(
    prompt,
    model='claude-3-5-sonnet-20240620',
    temperature=0,
    max_tokens=1024,
    max_retries=10_000,
    retry_delay=5,
):
    
    messages = [
        {"role": "user", "content": prompt},
    ]
    
    for attempt in range(max_retries):
        try:
            completion = anthropic_client.messages.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return completion.content[0].text
        except AnthropicRateLimitError:
            if attempt < max_retries - 1:  # If it's not the last attempt
                print(f"Rate limit reached. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise  # Re-raise the exception if we've exhausted all retries


from google.api_core import retry
from google.api_core import exceptions
import google.generativeai as genai

genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

def get_completion_response_gemini(
    prompt,
    model='gemini-flash',
    temperature=0,
    max_tokens=1024,
    max_retries=10_000,
    retry_delay=5,
):
    model = genai.GenerativeModel(model_name=model)
    
    @retry.Retry(
        predicate=lambda e: isinstance(e, exceptions.ResourceExhausted),
        initial=retry_delay,
        maximum=retry_delay * 10,
        multiplier=1.5,
        deadline=300.0,
    )
    def _generate_content():
        return model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        )

    for attempt in range(max_retries):
        try:
            response = _generate_content()
            return response.text
        except exceptions.ResourceExhausted:
            if attempt < max_retries - 1:  # If it's not the last attempt
                print(f"Rate limit reached. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise  # Re-raise the exception if we've exhausted all retries
                


def get_completion_response(
    prompt,
    **kwargs,
):
    model = kwargs.get('model', 'gpt-4o-mini')

    if model.startswith('claude'):
        if 'seed' in kwargs:
            kwargs.pop('seed')
        return get_completion_response_anthropic(prompt, **kwargs)
    elif model.startswith('gemini'):
        if 'seed' in kwargs:
            kwargs.pop('seed')
        return get_completion_response_gemini(prompt, **kwargs)
    else:
        return get_completion_response_openai(prompt, **kwargs)
