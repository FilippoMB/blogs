---
layout: single
title: Welcome to my page of tutorials
permalink: /
---

This is a collection of tutorials and blog posts about my interests and research topics.

<a id="tutorials"></a>

## Available tutorials

<ul>
{% assign docs = site.tutorials | sort: "nav_order" %}
{% for doc in docs %}
  <li><a href="{{ doc.url | relative_url }}">{{ doc.title }}</a></li>
{% endfor %}
</ul>
