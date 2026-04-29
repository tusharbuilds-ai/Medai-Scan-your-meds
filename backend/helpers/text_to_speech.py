from gemini_llm.gemini_llm import client
from google.genai import types
from logs.logger import logger
def text_to_speech(text:str)->bytes:
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name="Kore"
                        )
                    )
                )
            )
        )
        audio_data = response.candidates[0].content.parts[0].inline_data.mime_type
        return audio_data
    except Exception as error_in_text_to_speech:
        logger.error(f"Error -> {error_in_text_to_speech}")
        return " "
    
