// Netlify Function — proxies KRNO METAR from NOAA tgftp server
// Runs server-side so no CORS issues. Called by index.html as /.netlify/functions/metar

exports.handler = async function() {
    try {
        const response = await fetch('https://tgftp.nws.noaa.gov/data/observations/metar/stations/KRNO.TXT');
        if (!response.ok) throw new Error(`NOAA tgftp returned ${response.status}`);
        const text = await response.text();
        return {
            statusCode: 200,
            headers: {
                'Content-Type': 'text/plain',
                'Access-Control-Allow-Origin': '*',
                'Cache-Control': 'no-cache'
            },
            body: text
        };
    } catch(err) {
        return {
            statusCode: 500,
            headers: { 'Content-Type': 'text/plain' },
            body: `ERROR: ${err.message}`
        };
    }
};
