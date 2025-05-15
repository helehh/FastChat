import requests
import os

# Cloudflare Turnstile keys
TURNSTILE_SITE_KEY = os.environ.get("TURNSTILE_SITE_KEY")  # Site key
TURNSTILE_SECRET_KEY = os.environ.get("TURNSTILE_SECRET_KEY")  # Secret key

CLOUDFLARE_VERIFICATION_FAILED_MESSAGE = "PÄRINGU KONTROLL EBAÕNNESTUS. PALUN PROOVI UUESTI VÕI VÄRSKENDA LEHEKÜLGE!"


def verify_turnstile(token):
    """Verify the Turnstile token with Cloudflare"""
    url = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
    data = {"secret": TURNSTILE_SECRET_KEY, "response": token}

    response = requests.post(url, data=data)
    result = response.json()
    return result


cloudflare_turnstile_head_script = f"""
<script src="https://challenges.cloudflare.com/turnstile/v0/api.js" async defer></script>
<script defer>
    function onTurnstileLoad() {{
        window.turnstile.render('#turnstile-container', {{
            appearance: 'interaction-only',
            sitekey: '{TURNSTILE_SITE_KEY}',
            callback: function(token) {{
                // Find the hidden JSON component and set its value
                const hiddenInput = document.querySelector('#turnstile-token textarea');    
            
                hiddenInput.value = token;
                hiddenInput.dispatchEvent(new Event("input", {{ bubbles: true }}));

                console.log('cf token', token?.length )
                
                setTimeout(() => {{
                    document.querySelector('#turnstile-container').style.display = 'none'
                }}, 1000)
            }}
        }});
    }}
    
    // Check if turnstile is loaded
    let checkInterval = setInterval(function() {{
        if (window.turnstile && document.querySelector("#turnstile-container")) {{
            onTurnstileLoad();
            clearInterval(checkInterval);
        }}
    }}, 100);
</script>
"""
