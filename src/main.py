import os
import wave

from inference import SAMPLE_RATE, generate_tokens_from_api
from speechpipe import tokens_decoder_sync

# from inference import OrpheusCpp

# orpheus = OrpheusCpp()

text = """
Hey Dennis,

I was wondering if you'd be interested in participating in a panel discussion at the Cloud Summit.

A friend of mine, who is a co-organizer of the Cloud Summit, is putting together a panel focused on Cloud and AI.

The Cloud Summit is a significant event – I believe they had around 700 attendees last year (though don't quote me on that! :P). It's also sponsored by some major players, which could provide Unblocked with good visibility. Plus, I'm confident you'd bring a lot of valuable insights to the event. It's a win-win!

Please let me know if this is something you'd be interested in. The event is scheduled for May 27th."""

# # text = "I really hope the project deadline doesn't get moved up again."

# # output is a tuple of (sample_rate (24_000), samples (numpy int16 array))
# sample_rate, samples = orpheus.tts(text, options={"voice_id": "tara"})
# write("output.wav", sample_rate, samples.squeeze())

token_gen = generate_tokens_from_api(
    prompt=text,
    voice="tara",
)

samples = tokens_decoder_sync(token_gen)
# Create directory if it doesn't exist
os.makedirs(os.path.dirname(os.path.abspath("output.wav")), exist_ok=True)
wav_file = wave.open("output.wav", "wb")
wav_file.setnchannels(1)
wav_file.setsampwidth(2)
wav_file.setframerate(SAMPLE_RATE)

write_buffer = bytearray()
buffer_max_size = 1024 * 1024  # 1MB max buffer size (adjustable)

for audio in samples:
    write_buffer.extend(audio)

    # Flush buffer if it's large enough
    if len(write_buffer) >= buffer_max_size:
        wav_file.writeframes(write_buffer)
        write_buffer = bytearray()  # Reset buffer

if len(write_buffer) > 0:
    print(f"Final buffer flush: {len(write_buffer)} bytes")
    wav_file.writeframes(write_buffer)

# Close WAV file if opened
wav_file.close()
