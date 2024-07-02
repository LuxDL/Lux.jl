import{_ as s,c as a,o as n,a3 as i}from"./chunks/framework.Br70pw-E.js";const y=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),p={name:"tutorials/beginner/3_SimpleRNN.md"},l=i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, AMDGPU, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random,</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractExplicitContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/dev/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">       Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">AbstractExplicitContainerLayer{(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/dev/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/dev/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/dev/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56347</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51185</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.46991</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.44954</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.43293</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40996</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.37519</span></span>
<span class="line"><span>Validation: Loss 0.36849 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37533 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36529</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.34505</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33766</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.32692</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30689</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28266</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.26838</span></span>
<span class="line"><span>Validation: Loss 0.25844 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.26250 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.26695</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24311</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23449</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22133</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21033</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19772</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.18561</span></span>
<span class="line"><span>Validation: Loss 0.18050 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.18255 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.18034</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17275</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16406</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15480</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14891</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14175</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13642</span></span>
<span class="line"><span>Validation: Loss 0.12877 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.13022 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12721</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12363</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11574</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11316</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10982</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10263</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09638</span></span>
<span class="line"><span>Validation: Loss 0.09360 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09494 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09407</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08900</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08620</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08209</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07720</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07766</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07061</span></span>
<span class="line"><span>Validation: Loss 0.06904 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.07039 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07037</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06657</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06266</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06142</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05686</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05640</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05619</span></span>
<span class="line"><span>Validation: Loss 0.05149 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05278 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05087</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04888</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04844</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04746</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04329</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04225</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.03755</span></span>
<span class="line"><span>Validation: Loss 0.03870 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03989 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04066</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03687</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03770</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03308</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03276</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03086</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03075</span></span>
<span class="line"><span>Validation: Loss 0.02953 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03062 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02972</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02742</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02897</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02729</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02561</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02449</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02304</span></span>
<span class="line"><span>Validation: Loss 0.02320 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02418 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02415</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02206</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02194</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02185</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02057</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01943</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01916</span></span>
<span class="line"><span>Validation: Loss 0.01887 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01973 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01888</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01826</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01896</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01778</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01686</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01606</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01692</span></span>
<span class="line"><span>Validation: Loss 0.01586 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01661 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01564</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01652</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01381</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01571</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01640</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01316</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01209</span></span>
<span class="line"><span>Validation: Loss 0.01368 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01435 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01340</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01297</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01377</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01365</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01242</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01269</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01239</span></span>
<span class="line"><span>Validation: Loss 0.01206 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01265 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01158</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01204</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01192</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01125</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01182</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01116</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01167</span></span>
<span class="line"><span>Validation: Loss 0.01079 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01133 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01136</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01103</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00971</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00999</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01051</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01019</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01027</span></span>
<span class="line"><span>Validation: Loss 0.00975 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01025 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00975</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00990</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00944</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00986</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00945</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00847</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00969</span></span>
<span class="line"><span>Validation: Loss 0.00890 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00935 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00898</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00881</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00849</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00891</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00867</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00821</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00860</span></span>
<span class="line"><span>Validation: Loss 0.00817 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00859 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00836</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00766</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00783</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00850</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00765</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00813</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00700</span></span>
<span class="line"><span>Validation: Loss 0.00754 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00793 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00775</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00741</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00738</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00755</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00692</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00726</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00754</span></span>
<span class="line"><span>Validation: Loss 0.00700 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00736 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00733</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00715</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00703</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00689</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00639</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00652</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00650</span></span>
<span class="line"><span>Validation: Loss 0.00652 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00686 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00648</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00628</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00646</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00650</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00653</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00632</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00582</span></span>
<span class="line"><span>Validation: Loss 0.00610 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00642 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00662</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00583</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00648</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00561</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00598</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00566</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00531</span></span>
<span class="line"><span>Validation: Loss 0.00572 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00603 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00591</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00581</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00555</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00562</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00532</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00550</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00602</span></span>
<span class="line"><span>Validation: Loss 0.00539 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00567 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00554</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00515</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00564</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00538</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00501</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00505</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00566</span></span>
<span class="line"><span>Validation: Loss 0.00508 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00535 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56110</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50966</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47562</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.44407</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.42644</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.41980</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.39978</span></span>
<span class="line"><span>Validation: Loss 0.37687 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37726 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37056</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36058</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33344</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.32009</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30039</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28506</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.26894</span></span>
<span class="line"><span>Validation: Loss 0.26346 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.26375 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25978</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24103</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23934</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22599</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21240</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19893</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.18863</span></span>
<span class="line"><span>Validation: Loss 0.18390 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.18406 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.18221</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17713</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16380</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15642</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15038</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14178</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13804</span></span>
<span class="line"><span>Validation: Loss 0.13170 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.13179 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.13031</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12309</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12113</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11325</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10800</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10478</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10338</span></span>
<span class="line"><span>Validation: Loss 0.09648 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09655 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09388</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09268</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08737</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08483</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08040</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07655</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07081</span></span>
<span class="line"><span>Validation: Loss 0.07176 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.07180 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07527</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06762</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06283</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06184</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05794</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05765</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05355</span></span>
<span class="line"><span>Validation: Loss 0.05396 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05397 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05251</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05038</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04892</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04863</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04503</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04215</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.03762</span></span>
<span class="line"><span>Validation: Loss 0.04087 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04088 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04006</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03841</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03767</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03570</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03210</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03255</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03178</span></span>
<span class="line"><span>Validation: Loss 0.03140 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03141 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03005</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03087</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02910</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02670</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02413</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02634</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02289</span></span>
<span class="line"><span>Validation: Loss 0.02477 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02478 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02297</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02231</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02371</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02207</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02127</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02051</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01832</span></span>
<span class="line"><span>Validation: Loss 0.02020 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02023 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01882</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01970</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01913</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01857</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01641</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01619</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01726</span></span>
<span class="line"><span>Validation: Loss 0.01697 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01698 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01671</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01601</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01425</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01487</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01488</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01509</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01582</span></span>
<span class="line"><span>Validation: Loss 0.01463 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01465 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01423</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01379</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01331</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01292</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01279</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01330</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01144</span></span>
<span class="line"><span>Validation: Loss 0.01286 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01288 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01238</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01268</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01190</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01153</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01130</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01083</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01196</span></span>
<span class="line"><span>Validation: Loss 0.01149 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01150 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01160</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01086</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00988</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01061</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01076</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00982</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01008</span></span>
<span class="line"><span>Validation: Loss 0.01039 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01041 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01009</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00906</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00984</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00965</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00964</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00920</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00956</span></span>
<span class="line"><span>Validation: Loss 0.00947 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00948 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00949</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00860</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00854</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00934</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00841</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00834</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00796</span></span>
<span class="line"><span>Validation: Loss 0.00869 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00871 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00847</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00815</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00812</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00830</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00793</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00746</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00758</span></span>
<span class="line"><span>Validation: Loss 0.00802 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00804 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00777</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00746</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00728</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00750</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00746</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00716</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00747</span></span>
<span class="line"><span>Validation: Loss 0.00744 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00746 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00723</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00710</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00754</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00668</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00662</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00676</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00524</span></span>
<span class="line"><span>Validation: Loss 0.00693 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00695 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00639</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00674</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00669</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00660</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00628</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00603</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00626</span></span>
<span class="line"><span>Validation: Loss 0.00649 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00650 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00642</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00633</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00619</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00564</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00621</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00543</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00611</span></span>
<span class="line"><span>Validation: Loss 0.00609 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00610 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00583</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00547</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00576</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00585</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00565</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00547</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00556</span></span>
<span class="line"><span>Validation: Loss 0.00573 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00574 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00563</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00560</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00528</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00506</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00513</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00536</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00522</span></span>
<span class="line"><span>Validation: Loss 0.00540 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00541 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {compress </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">} ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>  JULIA_AMDGPU_LOGGING_ENABLED = true</span></span>
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
