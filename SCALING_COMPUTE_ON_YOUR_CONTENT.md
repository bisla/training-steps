
# Introducing Engram: Scaling compute on your context

**@EngramLab** · June 23, 2026 · [link](https://x.com/EngramLab/status/2069465879696576844)

We're Engram. We're building AI that learns from you and deeply understands your work.

Today's AI models don't understand what you do. Not really. Everything models know comes from their training – and they're trained mostly on the public internet. They're knowledgeable about popular Github repos and things people write in articles online.

But what you spend your time thinking about every day is so much more than that.

You know what good work looks like. You know where a specific project is going, and where you want it to be in a year. You know everything – from the small details, to the big picture of how life and work fit together – knowledge that goes far beyond any chat window. And when you do write things down, important ideas scatter across a sprawl of documents and files. When you use a model, it reads and re-reads many documents from your company before it can even get started.

But as we all know by now, this paradigm isn't perfect: as our contexts grow, models become more expensive and more confused. Individual users already generate far too much data for a model to process. And reading is shallow and temporary; even when the model does see your context, it forgets everything the moment you close the chat. Right now, models do not learn from this data. This means they can't automatically get better at the things you use them for.

We want to change this by building models that learn from your context.

Ours is a fundamentally different bet from other labs. Instead of spending massive amounts of training compute on public data, we start from strong pre-trained models and spend training compute on the context you care about. Each model spends the equivalent of hundreds of years studying your context: piecing things together, drawing connections that have never been drawn, finding errors that went unnoticed.

Through internal use and our design partnership with Notion, we've been training models that exhibit new and interesting types of behaviors. They've learned about us and our work from our GitHub, Slack, and Notion. They know about us, what we're working on, and why. They draw unexpected connections and remember things we forgot. For many tasks, our models don't need to re-gather context, so they can be 10x or even 100x more token-efficient. They just know things you'd expect your best teammate to know.

Our north star is a single training algorithm that can absorb arbitrary amounts of data into a model that gets continually better. We currently run this process on all of our company data every day, but are moving towards retraining every hour, and eventually, every minute. If this sounds interesting, we're hiring!

Despite all the buzz around continual learning, memory, and "learning from you", building a system like this that works (at scale, over many rounds of updates) is still an open problem. We know these problems are hard, because we've been working on them for years. Members of our team have worked on this problem from every angle: context compression, retrieval, LoRA, synthetic data, long-context and memory architectures. We've studied memorization and forgetting in humans and machines.

Our findings convinced us that we've identified a concrete new axis of scaling. Scaling compute to study and internalize data offers a tractable path to models that understand you and what you do.

Our first product is an API for agents that learn on very large shared knowledge workspaces. We're grateful to work with early partners that own some of the richest contexts and have been the earliest adopters of AI:

- With **@NotionHQ**, we're building Custom Agents that understand large Notion workspaces.
- With **@harvey**, we're developing models that internalize the knowledge of an entire firm and can search and find precedents across many client matters.
- With **@Microsoft**, we are piloting Engram models inside M365 to deliver cost-efficient, customized agents for their enterprise customers.

To pursue this vision, we've raised **$98M** from @generalcatalyst, @kleinerperkins, @sequoia, Factory, Modern, @AmplifyPartners, @neo, @svangel, and others. Our investors and advisors include Assaf Rappaport (@assaf_rappaport), Andrej Karpathy (@karpathy), and Pieter Abbeel (@pabbeel).

Interactions with future models will feel completely different. They'll generate trillions of tokens of context a day against a background of a constantly changing world. Every day, you'll teach things to a model – and have it actually learn. Your model.

