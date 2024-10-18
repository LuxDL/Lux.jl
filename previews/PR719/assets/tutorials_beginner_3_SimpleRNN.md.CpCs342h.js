import{_ as s,c as a,o as n,a3 as i}from"./chunks/framework.DODDq7nQ.js";const y=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),p={name:"tutorials/beginner/3_SimpleRNN.md"},l=i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, AMDGPU, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">      Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the spirals</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [MLUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Datasets</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">make_spiral</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(sequence_length) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> _ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dataset_size]</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the labels</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    labels </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">repeat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">repeat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    clockwise_spirals </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(d[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">][:, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">sequence_length], :, sequence_length, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">                         for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> d </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)]]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    anticlockwise_spirals </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                                 d[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">][:, (sequence_length </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], :, sequence_length, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">                             for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> d </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[((dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Float32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">cat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(clockwise_spirals</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, anticlockwise_spirals</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; dims</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Split the dataset</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (x_train, y_train), (x_val, y_val) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> splitobs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x_data, labels); at</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create DataLoaders</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Use DataLoader to automatically minibatch and shuffle the data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_train, y_train)); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Don&#39;t shuffle the validation data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_val, y_val)); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractExplicitContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/previews/PR719/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">       Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">AbstractExplicitContainerLayer{(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/previews/PR719/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/previews/PR719/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Main.var&quot;##225&quot;.SpiralClassifier</span></span></code></pre></div><p>We can use default Lux blocks – <code>Recurrence(LSTMCell(in_dims =&gt; hidden_dims)</code> – instead of defining the following. But let&#39;s still do it for the sake of it.</p><p>Now we need to define the behavior of the Classifier when it is invoked.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T, 3}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # First we will have to run the sequence through the LSTM Cell</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # The first call to LSTM Cell will create the initial hidden state</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # See that the parameters and states are automatically populated into a field called</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # \`lstm_cell\` We use \`eachslice\` to get the elements in the sequence without copying,</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # and \`Iterators.peel\` to split out the first element for LSTM initialization.</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_init, x_rest </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Iterators</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">peel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">_eachslice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (y, carry), st_lstm </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_init, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Now that we have the hidden state and memory in \`carry\` we will pass the input and</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # \`carry\` jointly</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x_rest</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        (y, carry), st_lstm </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, carry), ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell, st_lstm)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # After running through the sequence we will pass the output through the classifier</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y, st_classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">classifier, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">classifier)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Finally remember to create the updated state</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> merge</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st, (classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">st_classifier, lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">st_lstm))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vec</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y), st</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/previews/PR719/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; lstm_cell, classifier) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T, 3}</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x_init, x_rest </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Iterators</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">peel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">_eachslice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        y, carry </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_init)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x_rest</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y, carry </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, carry))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vec</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>SpiralClassifierCompact (generic function with 1 method)</span></span></code></pre></div><h2 id="Defining-Accuracy,-Loss-and-Optimiser" tabindex="-1">Defining Accuracy, Loss and Optimiser <a class="header-anchor" href="#Defining-Accuracy,-Loss-and-Optimiser" aria-label="Permalink to &quot;Defining Accuracy, Loss and Optimiser {#Defining-Accuracy,-Loss-and-Optimiser}&quot;">​</a></h2><p>Now let&#39;s define the binarycrossentropy loss. Typically it is recommended to use <code>logitbinarycrossentropy</code> since it is more numerically stable, but for the sake of simplicity we will use <code>binarycrossentropy</code>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> lossfn </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> BinaryCrossEntropyLoss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> compute_loss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, (x, y))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ŷ, st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, ps, st)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lossfn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss, st_, (; y_pred</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ŷ)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">matches</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((y_pred </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.5f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.==</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y_true)</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> matches</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>accuracy (generic function with 1 method)</span></span></code></pre></div><h2 id="Training-the-Model" tabindex="-1">Training the Model <a class="header-anchor" href="#Training-the-Model" aria-label="Permalink to &quot;Training the Model {#Training-the-Model}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model_type)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the dataloaders</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (train_loader, val_loader) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model_type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    rng </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Xoshiro</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Experimental</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        rng, model, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.01f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">); transform_variables</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dev)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">25</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Train the model</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> train_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            (_, loss, _, train_state) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Experimental</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">single_train_step!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                AutoZygote</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), lossfn, (x, y), train_state)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Epoch [%3d]: Loss %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch loss</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Validate the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> val_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ŷ, st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, st_)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lossfn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Validation: Loss %4.5f Accuracy %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss acc</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56247</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50855</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47341</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.45294</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.42439</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40220</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.38209</span></span>
<span class="line"><span>Validation: Loss 0.36584 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37201 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37659</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.34671</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33631</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30880</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.29721</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28532</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.27327</span></span>
<span class="line"><span>Validation: Loss 0.25603 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25964 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.26363</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24440</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23357</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21615</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20866</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19388</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19026</span></span>
<span class="line"><span>Validation: Loss 0.17850 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.18006 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17629</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16828</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16306</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15470</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14916</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14001</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13170</span></span>
<span class="line"><span>Validation: Loss 0.12702 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12807 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12734</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12240</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11538</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11138</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10498</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09994</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09775</span></span>
<span class="line"><span>Validation: Loss 0.09204 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09308 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09228</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08797</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08520</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07991</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07762</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07390</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07101</span></span>
<span class="line"><span>Validation: Loss 0.06769 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06883 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06804</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06587</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06282</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05913</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05652</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05417</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05664</span></span>
<span class="line"><span>Validation: Loss 0.05036 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05152 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04871</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04928</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04802</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04533</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04188</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04126</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04026</span></span>
<span class="line"><span>Validation: Loss 0.03781 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03891 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03894</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03621</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03552</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03305</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03361</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03044</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.02896</span></span>
<span class="line"><span>Validation: Loss 0.02885 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02988 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02912</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02838</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02628</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02709</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02583</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02355</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02264</span></span>
<span class="line"><span>Validation: Loss 0.02268 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02362 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02352</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02328</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02061</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02024</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01963</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02023</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01927</span></span>
<span class="line"><span>Validation: Loss 0.01848 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01931 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01805</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01856</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01798</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01663</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01664</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01736</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01566</span></span>
<span class="line"><span>Validation: Loss 0.01555 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01628 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01710</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01508</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01515</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01396</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01420</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01344</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01578</span></span>
<span class="line"><span>Validation: Loss 0.01343 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01407 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01330</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01488</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01303</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01240</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01183</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01247</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01186</span></span>
<span class="line"><span>Validation: Loss 0.01184 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01241 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01213</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01210</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01200</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01086</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01113</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01106</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.00972</span></span>
<span class="line"><span>Validation: Loss 0.01060 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01111 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01079</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00999</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01025</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00981</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01067</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01176</span></span>
<span class="line"><span>Validation: Loss 0.00960 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01007 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00971</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00993</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00938</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00903</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00985</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00877</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00773</span></span>
<span class="line"><span>Validation: Loss 0.00876 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00919 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00881</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00857</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00858</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00804</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00859</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00860</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00962</span></span>
<span class="line"><span>Validation: Loss 0.00805 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00846 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00851</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00777</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00826</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00850</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00745</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00704</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00759</span></span>
<span class="line"><span>Validation: Loss 0.00744 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00781 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00712</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00752</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00763</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00729</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00721</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00706</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00746</span></span>
<span class="line"><span>Validation: Loss 0.00691 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00726 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00699</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00681</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00657</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00673</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00667</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00694</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00718</span></span>
<span class="line"><span>Validation: Loss 0.00644 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00676 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00642</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00693</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00640</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00641</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00642</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00540</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00686</span></span>
<span class="line"><span>Validation: Loss 0.00602 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00633 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00600</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00617</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00574</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00578</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00607</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00601</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00545</span></span>
<span class="line"><span>Validation: Loss 0.00565 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00594 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00572</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00544</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00569</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00556</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00560</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00557</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00527</span></span>
<span class="line"><span>Validation: Loss 0.00531 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00559 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00572</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00502</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00525</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00540</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00506</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00524</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00481</span></span>
<span class="line"><span>Validation: Loss 0.00501 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00527 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56247</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51327</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47488</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.44122</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.42795</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40035</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.38217</span></span>
<span class="line"><span>Validation: Loss 0.36768 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37851 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37564</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.34356</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.32604</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31638</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30485</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28185</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.27349</span></span>
<span class="line"><span>Validation: Loss 0.25728 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.26454 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25500</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24076</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22597</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22107</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21180</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20061</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.18884</span></span>
<span class="line"><span>Validation: Loss 0.17889 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.18307 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17559</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17208</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16078</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15435</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14628</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14006</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13346</span></span>
<span class="line"><span>Validation: Loss 0.12694 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12964 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12730</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12099</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11674</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10736</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10513</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10073</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09887</span></span>
<span class="line"><span>Validation: Loss 0.09189 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09420 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09217</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08763</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08276</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08077</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07592</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07469</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07107</span></span>
<span class="line"><span>Validation: Loss 0.06759 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06978 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06869</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06407</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06287</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05792</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05633</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05532</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05117</span></span>
<span class="line"><span>Validation: Loss 0.05032 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05237 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05039</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04828</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04574</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04317</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04320</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04173</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.03867</span></span>
<span class="line"><span>Validation: Loss 0.03783 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03976 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03807</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03599</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03487</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03219</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03171</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03212</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03220</span></span>
<span class="line"><span>Validation: Loss 0.02891 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03068 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02717</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02818</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02693</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02601</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02547</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02420</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02474</span></span>
<span class="line"><span>Validation: Loss 0.02274 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02429 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02266</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02122</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02147</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01969</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02148</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01929</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01953</span></span>
<span class="line"><span>Validation: Loss 0.01851 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01984 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01818</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01898</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01701</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01769</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01716</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01496</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01506</span></span>
<span class="line"><span>Validation: Loss 0.01555 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01669 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01631</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01469</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01382</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01442</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01456</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01431</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01333</span></span>
<span class="line"><span>Validation: Loss 0.01344 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01444 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01335</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01266</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01278</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01301</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01293</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01208</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01140</span></span>
<span class="line"><span>Validation: Loss 0.01186 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01275 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01131</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01167</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01132</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01085</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01098</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01141</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01255</span></span>
<span class="line"><span>Validation: Loss 0.01062 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01143 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01047</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01050</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01040</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01010</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01051</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00938</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00895</span></span>
<span class="line"><span>Validation: Loss 0.00961 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01034 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00915</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00995</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00893</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00917</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00907</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00927</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00871</span></span>
<span class="line"><span>Validation: Loss 0.00878 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00945 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00884</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00839</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00800</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00889</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00820</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00849</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00822</span></span>
<span class="line"><span>Validation: Loss 0.00807 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00870 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00847</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00788</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00801</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00782</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00702</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00762</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00755</span></span>
<span class="line"><span>Validation: Loss 0.00746 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00804 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00771</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00742</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00704</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00736</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00719</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00668</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00680</span></span>
<span class="line"><span>Validation: Loss 0.00693 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00747 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00658</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00661</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00696</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00663</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00681</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00655</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00709</span></span>
<span class="line"><span>Validation: Loss 0.00646 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00697 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00676</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00643</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00586</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00641</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00591</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00631</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00592</span></span>
<span class="line"><span>Validation: Loss 0.00604 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00652 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00606</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00651</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00559</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00563</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00599</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00547</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00570</span></span>
<span class="line"><span>Validation: Loss 0.00567 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00612 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00543</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00555</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00587</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00515</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00576</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00545</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00493</span></span>
<span class="line"><span>Validation: Loss 0.00534 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00577 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00530</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00560</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00515</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00499</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00509</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00514</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00481</span></span>
<span class="line"><span>Validation: Loss 0.00504 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00544 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {compress </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">} ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
<span class="line"><span> :ps_trained</span></span>
<span class="line"><span> :st_trained</span></span></code></pre></div><h2 id="appendix" tabindex="-1">Appendix <a class="header-anchor" href="#appendix" aria-label="Permalink to &quot;Appendix&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">InteractiveUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxDeviceUtils)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(CUDA) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> LuxDeviceUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxCUDADevice)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        CUDA</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(AMDGPU) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> LuxDeviceUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxAMDGPUDevice)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        AMDGPU</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.10.4</span></span>
<span class="line"><span>Commit 48d4fd48430 (2024-06-04 10:41 UTC)</span></span>
<span class="line"><span>Build Info:</span></span>
<span class="line"><span>  Official https://julialang.org/ release</span></span>
<span class="line"><span>Platform Info:</span></span>
<span class="line"><span>  OS: Linux (x86_64-linux-gnu)</span></span>
<span class="line"><span>  CPU: 48 × AMD EPYC 7402 24-Core Processor</span></span>
<span class="line"><span>  WORD_SIZE: 64</span></span>
<span class="line"><span>  LIBM: libopenlibm</span></span>
<span class="line"><span>  LLVM: libLLVM-15.0.7 (ORCJIT, znver2)</span></span>
<span class="line"><span>Threads: 4 default, 0 interactive, 2 GC (on 2 virtual cores)</span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>  JULIA_CPU_THREADS = 2</span></span>
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span>
<span class="line"><span>  LD_LIBRARY_PATH = /usr/local/nvidia/lib:/usr/local/nvidia/lib64</span></span>
<span class="line"><span>  JULIA_PKG_SERVER = </span></span>
<span class="line"><span>  JULIA_NUM_THREADS = 4</span></span>
<span class="line"><span>  JULIA_CUDA_HARD_MEMORY_LIMIT = 100%</span></span>
<span class="line"><span>  JULIA_PKG_PRECOMPILE_AUTO = 0</span></span>
<span class="line"><span>  JULIA_DEBUG = Literate</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA runtime 12.5, artifact installation</span></span>
<span class="line"><span>CUDA driver 12.5</span></span>
<span class="line"><span>NVIDIA driver 555.42.2</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA libraries: </span></span>
<span class="line"><span>- CUBLAS: 12.5.2</span></span>
<span class="line"><span>- CURAND: 10.3.6</span></span>
<span class="line"><span>- CUFFT: 11.2.3</span></span>
<span class="line"><span>- CUSOLVER: 11.6.2</span></span>
<span class="line"><span>- CUSPARSE: 12.4.1</span></span>
<span class="line"><span>- CUPTI: 23.0.0</span></span>
<span class="line"><span>- NVML: 12.0.0+555.42.2</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Julia packages: </span></span>
<span class="line"><span>- CUDA: 5.4.2</span></span>
<span class="line"><span>- CUDA_Driver_jll: 0.9.0+0</span></span>
<span class="line"><span>- CUDA_Runtime_jll: 0.14.0+1</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Toolchain:</span></span>
<span class="line"><span>- Julia: 1.10.4</span></span>
<span class="line"><span>- LLVM: 15.0.7</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.453 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45),h=[l];function e(t,c,k,o,E,r){return n(),a("div",null,h)}const g=s(p,[["render",e]]);export{y as __pageData,g as default};
