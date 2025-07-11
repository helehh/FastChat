import requests


def verify_turnstile(token, turnstile_secret_key: str):
    """Verify the Turnstile token with Cloudflare"""
    if not turnstile_secret_key:
        raise ValueError("TURNSTILE_SECRET_KEY is not set")
    
    url = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
    data = {"secret": turnstile_secret_key, "response": token}

    response = requests.post(url, data=data)
    result = response.json()
    return result


def get_cloudflare_turnstile_head_script(turnstile_site_key: str):
    if not turnstile_site_key:
        raise ValueError("TURNSTILE_SITE_KEY is not set")
    
    return f"""
<script src="https://challenges.cloudflare.com/turnstile/v0/api.js" async defer></script>
<script defer>
    function onTurnstileLoad() {{
        window.turnstile.render('#turnstile-container', {{
            appearance: 'interaction-only',
            sitekey: '{turnstile_site_key}',
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
