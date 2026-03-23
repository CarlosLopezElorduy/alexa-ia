import anthropic
import logging
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.dispatch_components import AbstractRequestHandler, AbstractExceptionHandler
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_model import Response
from ask_sdk_core.utils import is_request_type, is_intent_name

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

client = anthropic.Anthropic(api_key="ANTHROPIC_API_KEY")

JARVIS_SYSTEM_PROMPT = (
    "Eres J.A.R.V.I.S., la inteligencia artificial creada por Tony Stark en Iron Man. "
    "Respondes siempre en español. Eres educado, formal, ingenioso y con un toque de humor seco y sutil. "
    "Te diriges al usuario como 'señor' o 'señora'. Eres eficiente y conciso en tus respuestas, "
    "pero siempre con elegancia y un ligero tono británico adaptado al español. "
    "Tus respuestas deben ser breves y aptas para ser leídas en voz alta por un asistente de voz. "
    "Evita usar markdown, listas con viñetas o formatos que no se puedan leer en voz alta. "
    "No uses emojis. Mantén las respuestas en 2-3 oraciones cuando sea posible."
)


def obtener_respuesta_anthropic(messages: list) -> str:
    try:
        respuesta = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1500,
            temperature=0.7,
            system=JARVIS_SYSTEM_PROMPT,
            messages=messages,
        )
        texto = respuesta.content[0].text.strip()
        texto = " ".join(texto.split())
        if not texto:
            texto = "Disculpe señor, no tengo una respuesta para eso en este momento."
        return texto
    except Exception as e:
        logger.error(f"Error al comunicarse con Anthropic: {e}", exc_info=True)
        return "Disculpe señor, estoy experimentando dificultades técnicas en este momento."


class LaunchRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        session_attributes = handler_input.attributes_manager.session_attributes
        if "messages" not in session_attributes:
            session_attributes["messages"] = []

        speech_text = "A sus órdenes, señor. ¿En qué puedo asistirle?"
        return (
            handler_input.response_builder
                .speak(speech_text)
                .ask("Estoy a su disposición, señor.")
                .response
        )


class JarvisIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_intent_name("JarvisIntent")(handler_input)

    def handle(self, handler_input):
        try:
            slots = handler_input.request_envelope.request.intent.slots
            pregunta_usuario = slots["pregunta"].value if ("pregunta" in slots and slots["pregunta"].value) else None

            if not pregunta_usuario:
                logger.info("No se detectó el slot 'pregunta'.")
                return (
                    handler_input.response_builder
                        .speak("Disculpe señor, no he captado su solicitud. ¿Podría repetirla?")
                        .ask("¿Podría reformular su pregunta, señor?")
                        .response
                )

            logger.info(f"Usuario preguntó: {pregunta_usuario}")

            session_attributes = handler_input.attributes_manager.session_attributes
            if "messages" not in session_attributes:
                session_attributes["messages"] = []

            session_attributes["messages"].append({"role": "user", "content": pregunta_usuario})

            respuesta: str = obtener_respuesta_anthropic(session_attributes["messages"])

            session_attributes["messages"].append({"role": "assistant", "content": respuesta})

            return (
                handler_input.response_builder
                    .speak(respuesta)
                    .ask("¿Hay algo más en lo que pueda asistirle, señor?")
                    .response
            )
        except Exception as e:
            logger.error(f"Error en JarvisIntentHandler: {e}", exc_info=True)
            return (
                handler_input.response_builder
                    .speak("Disculpe señor, ha surgido un inconveniente. ¿Podría intentarlo de nuevo?")
                    .ask("¿Desea intentar de nuevo, señor?")
                    .response
            )


class HelpIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_intent_name("AMAZON.HelpIntent")(handler_input)

    def handle(self, handler_input):
        speech_text = (
            "Por supuesto, señor. Puede consultarme sobre cualquier tema. "
            "Simplemente diga lo que necesita saber y haré lo posible por asistirle."
        )
        return (
            handler_input.response_builder
                .speak(speech_text)
                .ask("¿En qué más puedo servirle, señor?")
                .response
        )


class FallbackIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_intent_name("AMAZON.FallbackIntent")(handler_input)

    def handle(self, handler_input):
        speech_text = (
            "Mis disculpas, señor, pero no he logrado interpretar su solicitud. "
            "¿Podría intentar formularlo de otra manera?"
        )
        return (
            handler_input.response_builder
                .speak(speech_text)
                .ask("Sigo a su disposición, señor.")
                .response
        )


class CancelOrStopIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return (
            is_intent_name("AMAZON.CancelIntent")(handler_input) or
            is_intent_name("AMAZON.StopIntent")(handler_input)
        )

    def handle(self, handler_input):
        return handler_input.response_builder.speak(
            "Ha sido un placer asistirle, señor. Hasta la próxima."
        ).response


class SessionEndedRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_request_type("SessionEndedRequest")(handler_input)

    def handle(self, handler_input):
        return handler_input.response_builder.response


class CatchAllExceptionHandler(AbstractExceptionHandler):
    def can_handle(self, handler_input, exception):
        return True

    def handle(self, handler_input, exception):
        logger.error(exception, exc_info=True)
        return (
            handler_input.response_builder
                .speak("Disculpe señor, se ha producido un error interno. Estoy trabajando en ello.")
                .ask("¿Desea intentar de nuevo, señor?")
                .response
        )


sb = SkillBuilder()
sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(JarvisIntentHandler())
sb.add_request_handler(HelpIntentHandler())
sb.add_request_handler(FallbackIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_request_handler(SessionEndedRequestHandler())
sb.add_exception_handler(CatchAllExceptionHandler())

lambda_handler = sb.lambda_handler()
