from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse
from twilio.twiml.messaging_response import MessagingResponse
from src.core.services.messaging_service import handle_whatsapp_message

router = APIRouter()

@router.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    form = await request.form()
    incoming_msg = (form.get("Body") or "").strip()
    from_number = (form.get("From") or "").strip()

    reply_text = handle_whatsapp_message(incoming_msg, from_number)

    twilio_resp = MessagingResponse()
    twilio_resp.message(reply_text)
    return PlainTextResponse(str(twilio_resp), media_type="application/xml")
