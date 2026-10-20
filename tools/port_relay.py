"""TCP ingress only: exposes the isolated API without giving it outbound routing."""
import asyncio

async def forward(reader,writer):
    try:
        while data:=await reader.read(65536):
            writer.write(data);await writer.drain()
    finally:
        writer.close()

async def connection(reader,writer):
    upstream=None
    try:
        incoming,upstream=await asyncio.wait_for(asyncio.open_connection('api',18761),10)
        await asyncio.gather(forward(reader,upstream),forward(incoming,writer))
    except (OSError,TimeoutError,ConnectionError):
        writer.close()
        if upstream:upstream.close()

async def main():
    server=await asyncio.start_server(connection,'0.0.0.0',18761)
    async with server:await server.serve_forever()

if __name__=='__main__':asyncio.run(main())
