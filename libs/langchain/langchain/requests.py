"""Lightweight wrapper around requests library, with async support."""
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional
from bs4 import BeautifulSoup

import aiohttp
import requests
from pydantic import BaseModel, Extra


class Requests(BaseModel):
    """Wrapper around requests to handle auth and async.

    The main purpose of this wrapper is to handle authentication (by saving
    headers) and enable easy async methods on the same base object.
    """

    headers: Optional[Dict[str, str]] = None
    aiosession: Optional[aiohttp.ClientSession] = None
    auth: Optional[Any] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        """GET the URL and return the text."""
        return requests.get(url, headers=self.headers, auth=self.auth, **kwargs)

    def post(self, url: str, data: Dict[str, Any], **kwargs: Any) -> requests.Response:
        """POST to the URL and return the text."""
        return requests.post(
            url, json=data, headers=self.headers, auth=self.auth, **kwargs
        )

    def patch(self, url: str, data: Dict[str, Any], **kwargs: Any) -> requests.Response:
        """PATCH the URL and return the text."""
        return requests.patch(
            url, json=data, headers=self.headers, auth=self.auth, **kwargs
        )

    def put(self, url: str, data: Dict[str, Any], **kwargs: Any) -> requests.Response:
        """PUT the URL and return the text."""
        return requests.put(
            url, json=data, headers=self.headers, auth=self.auth, **kwargs
        )

    def delete(self, url: str, **kwargs: Any) -> requests.Response:
        """DELETE the URL and return the text."""
        return requests.delete(url, headers=self.headers, auth=self.auth, **kwargs)

    @asynccontextmanager
    async def _arequest(
        self, method: str, url: str, **kwargs: Any
    ) -> AsyncGenerator[aiohttp.ClientResponse, None]:
        """Make an async request."""
        if not self.aiosession:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method, url, headers=self.headers, auth=self.auth, **kwargs
                ) as response:
                    yield response
        else:
            async with self.aiosession.request(
                method, url, headers=self.headers, auth=self.auth, **kwargs
            ) as response:
                yield response

    @asynccontextmanager
    async def aget(
        self, url: str, **kwargs: Any
    ) -> AsyncGenerator[aiohttp.ClientResponse, None]:
        """GET the URL and return the text asynchronously."""
        async with self._arequest("GET", url, auth=self.auth, **kwargs) as response:
            yield response

    @asynccontextmanager
    async def apost(
        self, url: str, data: Dict[str, Any], **kwargs: Any
    ) -> AsyncGenerator[aiohttp.ClientResponse, None]:
        """POST to the URL and return the text asynchronously."""
        async with self._arequest(
            "POST", url, json=data, auth=self.auth, **kwargs
        ) as response:
            yield response

    @asynccontextmanager
    async def apatch(
        self, url: str, data: Dict[str, Any], **kwargs: Any
    ) -> AsyncGenerator[aiohttp.ClientResponse, None]:
        """PATCH the URL and return the text asynchronously."""
        async with self._arequest(
            "PATCH", url, json=data, auth=self.auth, **kwargs
        ) as response:
            yield response

    @asynccontextmanager
    async def aput(
        self, url: str, data: Dict[str, Any], **kwargs: Any
    ) -> AsyncGenerator[aiohttp.ClientResponse, None]:
        """PUT the URL and return the text asynchronously."""
        async with self._arequest(
            "PUT", url, json=data, auth=self.auth, **kwargs
        ) as response:
            yield response

    @asynccontextmanager
    async def adelete(
        self, url: str, **kwargs: Any
    ) -> AsyncGenerator[aiohttp.ClientResponse, None]:
        """DELETE the URL and return the text asynchronously."""
        async with self._arequest("DELETE", url, auth=self.auth, **kwargs) as response:
            yield response

def _clean_url(url: str) -> str:
    """Strips quotes from the url."""
    return url.strip("\"'")

class TextRequestsWrapper(BaseModel):
    """Lightweight wrapper around requests library.

    The main purpose of this wrapper is to always return a text output.
    """

    headers: Optional[Dict[str, str]] = None
    aiosession: Optional[aiohttp.ClientSession] = None
    auth: Optional[Any] = None

    

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def requests(self) -> Requests:
        return Requests(
            headers=self.headers, aiosession=self.aiosession, auth=self.auth
        )

    def get(self, url: str, **kwargs: Any) -> str:
        """GET the URL and return the text."""
        return self.requests.get(url, **kwargs).text

    def post(self, url: str, data: Dict[str, Any], **kwargs: Any) -> str:
        """POST to the URL and return the text."""
        return self.requests.post(url, data, **kwargs).text

    def patch(self, url: str, data: Dict[str, Any], **kwargs: Any) -> str:
        """PATCH the URL and return the text."""
        return self.requests.patch(url, data, **kwargs).text

    def put(self, url: str, data: Dict[str, Any], **kwargs: Any) -> str:
        """PUT the URL and return the text."""
        return self.requests.put(url, data, **kwargs).text

    def delete(self, url: str, **kwargs: Any) -> str:
        """DELETE the URL and return the text."""
        return self.requests.delete(url, **kwargs).text
    
    def get_multi_text(self, urls:str, **kwargs: Any ) -> list:
        """get page text of multiple urls and concatnate all"""
        url_list = urls.split(',')
        html_codes = [self.requests.get(_clean_url(url),**kwargs).text for url in url_list]
        text_list = [BeautifulSoup(html_code, 'html.parser').get_text() for html_code in html_codes]
        return text_list
        
    async def aget_multi_text(self, urls:str, **kwargs: Any ) -> list:
        """run the tool asynchronously"""
        url_list = urls.split(',')
        html_codes = [self.requests.aget(_clean_url(url),**kwargs).text() for url in url_list]
        text_list = [BeautifulSoup(html_code, 'html.parser').get_text() for html_code in html_codes]
        return await text_list


    async def aget(self, url: str, **kwargs: Any) -> str:
        """GET the URL and return the text asynchronously."""
        async with self.requests.aget(url, **kwargs) as response:
            return await response.text()

    async def apost(self, url: str, data: Dict[str, Any], **kwargs: Any) -> str:
        """POST to the URL and return the text asynchronously."""
        async with self.requests.apost(url, data, **kwargs) as response:
            return await response.text()

    async def apatch(self, url: str, data: Dict[str, Any], **kwargs: Any) -> str:
        """PATCH the URL and return the text asynchronously."""
        async with self.requests.apatch(url, data, **kwargs) as response:
            return await response.text()

    async def aput(self, url: str, data: Dict[str, Any], **kwargs: Any) -> str:
        """PUT the URL and return the text asynchronously."""
        async with self.requests.aput(url, data, **kwargs) as response:
            return await response.text()

    async def adelete(self, url: str, **kwargs: Any) -> str:
        """DELETE the URL and return the text asynchronously."""
        async with self.requests.adelete(url, **kwargs) as response:
            return await response.text()


# For backwards compatibility
RequestsWrapper = TextRequestsWrapper
